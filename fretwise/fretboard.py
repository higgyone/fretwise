"""Stage 6 - fretboard mapping.

A pitch can be played in several places on a guitar. This maps each note to
every position that produces it, then picks one, preferring positions that
keep the hand where it already is rather than sending it up and down the neck
between consecutive notes.

Notes struck together are one chord and divide the strings between them, so
a chord is assigned as a whole rather than note by note.

A note may take a string that an earlier note is still sounding on. That is
not a compromise but what the instrument does: fretting a string stops
whatever was ringing on it, and the earlier note is shortened to match.
Preferring a free string is a cost rather than a rule, because transcribed
durations run longer than a guitar sustains, and treating them as binding
sent the hand up the neck to avoid collisions a player would never have had.

Assignment is greedy in time order and cannot revisit a choice, so it is
not guaranteed optimal -- only playable. Anything it still cannot place
leaves ``chosen`` as ``None`` and is counted rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from .notes import name_to_midi

# Standard tuning, thinnest string first. String 1 is the high E, string 6 the
# low E, matching how guitarists number them.
STANDARD_TUNING = ("E4", "B3", "G3", "D3", "A2", "E2")
DEFAULT_MAX_FRET = 22

# An open string needs no hand position, so it costs nothing to reach.
OPEN_STRING = 0

# Fretting a string stops whatever was ringing on it, so a note may take a
# string another note is still sounding on -- that note simply ends there.
# The cost is expressed in frets of hand movement: at 4, a player will cut a
# ringing note short rather than shift more than four frets to avoid it, which
# is the trade a guitarist actually makes. Measured on a strummed part, this
# keeps the whole piece below the ninth fret; treating a ringing string as
# untouchable instead pushed notes to the fourteenth, and left two of them
# with nowhere to go at all.
STEAL_COST = 4

# Notes struck this close together are one chord. They share the hand, so they
# divide the strings between them rather than taking them from each other.
CHORD_WINDOW = 0.06


@dataclass(frozen=True)
class Position:
    """Somewhere a note can be played."""

    string: int  # 1 = thinnest
    fret: int

    def to_dict(self) -> dict:
        return {"string": self.string, "fret": self.fret}


def tuning_midi(tuning: tuple[str, ...] = STANDARD_TUNING) -> list[int]:
    """MIDI number of each open string."""
    return [name_to_midi(name) for name in tuning]


def positions(
    midi: int,
    *,
    tuning: tuple[str, ...] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
) -> list[Position]:
    """Every way to play a pitch, ordered from the thinnest string down."""
    found = []
    for index, open_midi in enumerate(tuning_midi(tuning), start=1):
        fret = midi - open_midi
        if OPEN_STRING <= fret <= max_fret:
            found.append(Position(string=index, fret=fret))
    return found


def reach_cost(position: Position, anchor: int | None) -> tuple[int, int, int]:
    """How awkward this position is, given where the hand already is.

    Sorted lowest-first, so the tuple orders by hand movement, then by
    staying near the nut, then by string number to break remaining ties.
    """
    if position.fret == OPEN_STRING or anchor is None:
        movement = 0
    else:
        movement = abs(position.fret - anchor)
    return movement, position.fret, position.string


def choose(
    midis: list[int],
    *,
    anchor: int | None = None,
    taken: set[int] | None = None,
    ringing: set[int] | None = None,
    steal_cost: int = STEAL_COST,
    tuning: tuple[str, ...] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
) -> dict[int, Position | None]:
    """Assign a position to each of several pitches sounding together.

    ``taken`` strings are unavailable: another note of this same chord is on
    them. ``ringing`` strings carry a note still sounding from earlier, and
    may be used at the price of ``steal_cost`` -- fretting stops that note.

    Pitches with the fewest options are placed first, so a note that can only
    be played in one place is not left stranded by an earlier, freer note
    taking its string.
    """
    taken = set(taken or ())
    ringing = set(ringing or ())
    options = {
        midi: positions(midi, tuning=tuning, max_fret=max_fret) for midi in midis
    }

    def cost(position: Position) -> tuple[int, int, int]:
        movement, fret, string = reach_cost(position, anchor)
        return movement + (steal_cost if position.string in ringing else 0), fret, string

    chosen: dict[int, Position | None] = {}
    for midi in sorted(midis, key=lambda m: len(options[m])):
        available = [p for p in options[midi] if p.string not in taken]
        if not available:
            chosen[midi] = None  # no string left free; the chord cannot be voiced
            continue
        pick = min(available, key=cost)
        chosen[midi] = pick
        taken.add(pick.string)
    return chosen


def group_chords(notes, *, chord_window: float = CHORD_WINDOW) -> list[list]:
    """Notes struck close enough together to be one grab of the hand."""
    groups: list[list] = []
    for note in sorted(notes, key=lambda n: (n.time, n.midi)):
        if groups and note.time - groups[-1][0].time <= chord_window:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def map_notes(
    notes,
    *,
    tuning: tuple[str, ...] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
    steal_cost: int = STEAL_COST,
    chord_window: float = CHORD_WINDOW,
    stats: dict | None = None,
) -> list:
    """Fill in each note's playable positions and the one chosen.

    Walks the notes in time order. A string carrying a note that is still
    sounding can still be used -- fretting it stops that note, which is what
    happens on the instrument -- so the note that was ringing is shortened to
    end where the new one begins. Preferring a free string is expressed as a
    cost rather than a rule, so the hand is not sent up the neck to avoid
    cutting a note short.

    ``stats`` if given collects counts worth reporting: ``shortened`` and
    ``unplaced``.
    """
    ordered = sorted(notes, key=lambda n: (n.time, n.midi))
    for note in ordered:
        note.options = [
            p.to_dict() for p in positions(note.midi, tuning=tuning, max_fret=max_fret)
        ]

    anchor: int | None = None
    sounding: dict[int, object] = {}  # string -> the note still ringing on it
    shortened = unplaced = 0

    for group in group_chords(ordered, chord_window=chord_window):
        start_time = group[0].time
        for string, note in list(sounding.items()):
            if note.time + note.duration <= start_time:
                del sounding[string]

        picks = choose(
            [n.midi for n in group],
            anchor=anchor,
            ringing=set(sounding),
            steal_cost=steal_cost,
            tuning=tuning,
            max_fret=max_fret,
        )

        for note in group:
            pick = picks.get(note.midi)
            note.chosen = pick.to_dict() if pick else None
            if pick is None:
                unplaced += 1
                continue

            victim = sounding.get(pick.string)
            if victim is not None and victim is not note:
                # Fretting the string stops it: the earlier note ends here.
                victim.duration = max(0.0, note.time - victim.time)
                shortened += 1

            sounding[pick.string] = note
            if pick.fret != OPEN_STRING:
                anchor = pick.fret

    if stats is not None:
        stats.update(shortened=shortened, unplaced=unplaced)
    return ordered
