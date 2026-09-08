"""ASCII tab export.

Renders the mapped notes as guitar tablature: six lines, one per string,
with fret numbers where notes are played.

Spacing is proportional to elapsed time rather than laid out on a fixed grid.
A grid looks tidier but has to round every onset to the nearest column, and on
real transcribed material that silently merges notes played close together --
measured on a strummed part, a 0.15s grid lost 12% of the events outright.
Proportional spacing keeps every note, with rests capped so that a few seconds
of silence does not push the rest of the phrase off the page.

There is no beat detection here, so this is not rhythmically notated tab. It
shows what to play, in order, with the gaps roughly to scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fretboard import STANDARD_TUNING
from .timestamps import format_timestamp

# Notes closer together than this are one strum, and share a column.
CHORD_WINDOW = 0.06
# How much time one column stands for, before capping.
SECONDS_PER_COLUMN = 0.15
# The most columns a gap may take, however long the silence.
MAX_REST_COLUMNS = 8
# Printable width of one system, excluding the string labels.
SYSTEM_WIDTH = 76

# Tab names the strings by letter, the highest in lower case.
STRING_LABELS = ("e", "B", "G", "D", "A", "E")


@dataclass
class Event:
    """Notes struck at the same moment."""

    time: float
    notes: list


def group_events(notes, *, chord_window: float = CHORD_WINDOW) -> list[Event]:
    """Collect notes into simultaneous events, earliest first."""
    events: list[Event] = []
    for note in sorted(notes, key=lambda n: n.time):
        if events and note.time - events[-1].time <= chord_window:
            events[-1].notes.append(note)
        else:
            events.append(Event(time=note.time, notes=[note]))
    return events


def column_gaps(
    events: list[Event],
    *,
    seconds_per_column: float = SECONDS_PER_COLUMN,
    max_rest: int = MAX_REST_COLUMNS,
    min_gap: int = 2,
) -> list[int]:
    """How many columns to leave before each event after the first."""
    gaps = []
    for earlier, later in zip(events, events[1:]):
        columns = round((later.time - earlier.time) / seconds_per_column)
        gaps.append(max(min_gap, min(int(columns), max_rest)))
    return gaps


def place(events: list[Event], gaps: list[int]) -> tuple[list[list[str]], list[tuple[int, float]]]:
    """Lay the events out on six rows of characters.

    Returns the rows and, for labelling, the column each event landed in
    paired with its time.
    """
    rows = [[] for _ in STRING_LABELS]
    marks: list[tuple[int, float]] = []
    column = 0

    for index, event in enumerate(events):
        if index:
            column += gaps[index - 1]

        placed = {}
        for note in event.notes:
            if note.chosen is None:
                continue  # no string was free for it; nothing to draw
            placed[note.chosen["string"]] = str(note.chosen["fret"])

        if not placed:
            continue
        marks.append((column, event.time))

        width = max(len(text) for text in placed.values())
        for row_index, row in enumerate(rows):
            while len(row) < column:
                row.append("-")
            text = placed.get(row_index + 1, "")
            row.extend(text.ljust(width, "-") if text else "-" * width)

        # Leave room so the next event cannot run into a two-digit fret.
        column += width

    longest = max((len(row) for row in rows), default=0)
    for row in rows:
        row.extend("-" * (longest - len(row)))
    return rows, marks


def marker_line(
    marks: list[tuple[int, float]],
    start: int,
    width: int,
    indent: int,
    notable: set[int],
) -> str:
    """A line of timestamps above a system, at the columns they belong to.

    Only the pauses are labelled, plus wherever a system begins, so the line
    stays readable. Labelling every event turns it into a wall of digits.
    """
    here = [(column, time) for column, time in marks if start <= column < start + width]
    if not here:
        return ""

    wanted = [pair for pair in here if pair[0] in notable]
    if not wanted:
        wanted = [here[0]]  # always say where a system starts
    elif wanted[0] != here[0]:
        wanted.insert(0, here[0])

    line = [" "] * (indent + width)
    for column, time in wanted:
        label = format_timestamp(time)[:-2]  # tenths are enough here
        at = indent + column - start
        if at + len(label) > len(line):
            continue  # would run off the end of the system
        # Keep a space either side, or two labels read as one number.
        room = line[max(0, at - 1) : at + len(label) + 1]
        if all(character == " " for character in room):
            line[at : at + len(label)] = label
    return "".join(line).rstrip()


def render(
    notes,
    *,
    tuning: tuple[str, ...] = STANDARD_TUNING,
    seconds_per_column: float = SECONDS_PER_COLUMN,
    max_rest: int = MAX_REST_COLUMNS,
    width: int = SYSTEM_WIDTH,
    chord_window: float = CHORD_WINDOW,
) -> str:
    """Render notes as ASCII tab."""
    playable = [n for n in notes if n.chosen is not None]
    if not playable:
        return "no notes to write\n"

    events = group_events(playable, chord_window=chord_window)
    gaps = column_gaps(
        events, seconds_per_column=seconds_per_column, max_rest=max_rest
    )
    rows, marks = place(events, gaps)

    # A pause worth naming: anything that took at least half the rest cap.
    pause = max(max_rest // 2, 2)
    notable = {
        column
        for (column, _time), gap in zip(marks[1:], gaps)
        if gap >= pause
    }

    labels = [f"{name}|" for name in STRING_LABELS]
    indent = max(len(label) for label in labels)
    total = len(rows[0])

    lines = []
    for start in range(0, total, width):
        lines.append(marker_line(marks, start, width, indent, notable))
        for label, row in zip(labels, rows):
            lines.append(label.ljust(indent) + "".join(row[start : start + width]))
        lines.append("")

    return "\n".join(lines)


def header(notes, key: dict | None, *, title: str = "") -> str:
    """A few lines of context above the tab itself."""
    playable = sum(1 for n in notes if n.chosen is not None)
    skipped = len(notes) - playable

    lines = [line for line in (title,) if line]
    lines.append(
        f"{playable} notes"
        + (f" in {key['name']}" if key else "")
        + (f"  ({skipped} with no string free, not shown)" if skipped else "")
    )
    lines.append(
        "Spacing is proportional to time, not beats: there is no rhythm "
        "detection here."
    )
    lines.append("")
    return "\n".join(lines)
