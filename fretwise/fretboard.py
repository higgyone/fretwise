"""Stage 6 - fretboard mapping.

A pitch can be played in several places on a guitar. This maps each note to
every position that produces it, then picks one, preferring positions that
keep the hand where it already is rather than sending it up and down the neck
between consecutive notes.

Notes that sound at the same time cannot share a string, so a chord is
assigned as a whole rather than note by note.

Assignment is greedy in time order, which occasionally strands a note: an
earlier note with several homes may take the one string a later note
depends on, and the choice cannot be taken back once made. A one-note
lookahead covers the common case. What remains is reported rather than
hidden -- `map_notes` leaves ``chosen`` as ``None``, and the analyze
command counts them. Transcribed durations also overlap more freely than a
guitar allows, since fretting a string silences whatever was ringing on it,
so some of these collisions are not real.
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
    tuning: tuple[str, ...] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
) -> dict[int, Position | None]:
    """Assign a position to each of several pitches sounding together.

    Pitches with the fewest options are placed first, so a note that can only
    be played in one place is not left stranded by an earlier, freer note
    taking its string.
    """
    taken = set(taken or ())
    options = {
        midi: positions(midi, tuning=tuning, max_fret=max_fret) for midi in midis
    }

    chosen: dict[int, Position | None] = {}
    for midi in sorted(midis, key=lambda m: len(options[m])):
        available = [p for p in options[midi] if p.string not in taken]
        if not available:
            chosen[midi] = None  # no string left free; the chord cannot be voiced
            continue
        pick = min(available, key=lambda p: reach_cost(p, anchor))
        chosen[midi] = pick
        taken.add(pick.string)
    return chosen


def strings_needed_soon(
    notes,
    index: int,
    *,
    tuning: tuple[str, ...] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
) -> set[int]:
    """Strings that notes beginning during this one can only be played on.

    A guitarist does not occupy the low E while a note that needs it is
    about to sound. Without this, a note with several homes takes the one
    string a later, more constrained note depends on, stranding it.
    """
    note = notes[index]
    end = note.time + note.duration

    reserved = set()
    for later in notes[index + 1 :]:
        if later.time >= end:
            break
        elsewhere = positions(later.midi, tuning=tuning, max_fret=max_fret)
        if len(elsewhere) == 1:
            reserved.add(elsewhere[0].string)
    return reserved


def map_notes(
    notes,
    *,
    tuning: tuple[str, ...] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
) -> list:
    """Fill in each note's playable positions and the one chosen.

    Walks the notes in time order, keeping track of which strings are still
    ringing so simultaneous notes are not assigned to the same one, and
    leaving free any string a note starting shortly can only use.
    """
    ordered = sorted(notes, key=lambda n: (n.time, n.midi))
    anchor: int | None = None
    sounding: list = []  # (end_time, string) for notes still ringing

    for index, note in enumerate(ordered):
        note.options = [
            p.to_dict() for p in positions(note.midi, tuning=tuning, max_fret=max_fret)
        ]

        sounding = [(end, string) for end, string in sounding if end > note.time]
        taken = {string for _end, string in sounding}
        reserved = strings_needed_soon(
            ordered, index, tuning=tuning, max_fret=max_fret
        )

        pick = choose(
            [note.midi], anchor=anchor, taken=taken | reserved,
            tuning=tuning, max_fret=max_fret,
        )[note.midi]
        if pick is None:  # only a reserved string is left; take it rather than nothing
            pick = choose(
                [note.midi], anchor=anchor, taken=taken,
                tuning=tuning, max_fret=max_fret,
            )[note.midi]

        note.chosen = pick.to_dict() if pick else None
        if pick is not None:
            sounding.append((note.time + note.duration, pick.string))
            if pick.fret != OPEN_STRING:
                anchor = pick.fret

    return ordered
