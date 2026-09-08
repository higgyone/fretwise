"""ASCII tab export."""

import pytest

from fretwise import tab
from fretwise.analyze import DetectedNote
from fretwise.notes import midi_to_hz, midi_to_name, name_to_midi


def note(name, time, *, string, fret, duration=0.3, confidence=0.6):
    midi = name_to_midi(name)
    made = DetectedNote(
        time=time, duration=duration, note=midi_to_name(midi), midi=midi,
        hz=midi_to_hz(midi), cents=0.0, confidence=confidence,
    )
    made.chosen = None if string is None else {"string": string, "fret": fret}
    return made


def rows_of(text):
    """The six string rows of the first system, without their labels."""
    lines = [line for line in text.splitlines() if line[1:2] == "|"]
    return [line[2:] for line in lines[:6]]


def test_a_note_lands_on_its_own_string():
    text = tab.render([note("D3", 0.0, string=4, fret=0)])
    rows = rows_of(text)
    assert rows[3].startswith("0")  # string 4 is the D, fourth row down
    assert set("".join(rows[:3]) + "".join(rows[4:])) == {"-"}


def test_the_rows_are_labelled_high_string_first():
    text = tab.render([note("E2", 0.0, string=6, fret=0)])
    labels = [line[0] for line in text.splitlines() if line[1:2] == "|"]
    assert labels[:6] == ["e", "B", "G", "D", "A", "E"]


def test_a_chord_shares_one_column():
    """Notes struck together line up vertically, as a player reads them."""
    chord = [
        note("D3", 0.0, string=4, fret=0),
        note("A3", 0.01, string=3, fret=2),
        note("F#4", 0.02, string=1, fret=2),
    ]
    rows = rows_of(tab.render(chord))
    assert rows[0][0] == "2" and rows[2][0] == "2" and rows[3][0] == "0"


def test_notes_played_apart_do_not_share_a_column():
    spread = [
        note("D3", 0.0, string=4, fret=0),
        note("E3", 1.0, string=4, fret=2),
    ]
    row = rows_of(tab.render(spread))[3]
    assert row.index("0") < row.index("2")


def test_a_two_digit_fret_does_not_collide_with_the_next_note():
    notes = [
        note("C4", 0.0, string=4, fret=10),
        note("D3", 0.3, string=4, fret=0),
    ]
    row = rows_of(tab.render(notes))[3]
    assert row.startswith("10")
    assert "100" not in row  # the following 0 must not read as part of the 10


def test_a_long_silence_is_capped():
    """A minute of rest must not push the next phrase off the page."""
    notes = [
        note("D3", 0.0, string=4, fret=0),
        note("D3", 60.0, string=4, fret=0),
    ]
    row = rows_of(tab.render(notes, max_rest=6))[3]
    assert len(row) < 20


def test_gaps_never_collapse_two_events_together():
    """Every event keeps its own column, however fast the playing."""
    notes = [note("D3", i * 0.02, string=4, fret=i % 5) for i in range(6)]
    events = tab.group_events(notes, chord_window=0.001)
    gaps = tab.column_gaps(events)
    assert all(gap >= 2 for gap in gaps)


def test_notes_with_no_string_are_skipped_and_counted():
    notes = [note("D3", 0.0, string=4, fret=0), note("G2", 0.5, string=None, fret=0)]
    assert "1 with no string free" in tab.header(notes, None)
    assert rows_of(tab.render(notes))[3].count("0") == 1


def test_no_notes_at_all():
    assert "no notes" in tab.render([])


def test_wrapping_into_systems():
    notes = [note("D3", i * 0.5, string=4, fret=1) for i in range(40)]
    text = tab.render(notes, width=40)
    systems = [line for line in text.splitlines() if line.startswith("D|")]
    assert len(systems) > 1
    assert all(len(line) <= 42 for line in systems)


def test_timestamps_do_not_run_into_each_other():
    """Two labels with no gap between read as one meaningless number."""
    notes = [note("D3", i * 1.4, string=4, fret=0) for i in range(12)]
    text = tab.render(notes)
    for line in text.splitlines():
        if ":" not in line:
            continue
        for piece in line.split():
            assert piece.count(":") == 1, f"labels ran together: {line!r}"


def test_timestamps_stay_inside_the_system():
    notes = [note("D3", i * 1.3, string=4, fret=0) for i in range(30)]
    text = tab.render(notes, width=30)
    for line in text.splitlines():
        assert len(line) <= 34, f"line overflows: {line!r}"


def test_the_header_says_the_spacing_is_not_rhythmic():
    """The one thing a reader could reasonably misread it as."""
    assert "not beats" in tab.header([note("D3", 0.0, string=4, fret=0)], None)


def test_the_header_names_the_key_when_there_is_one():
    text = tab.header([note("D3", 0.0, string=4, fret=0)], {"name": "D major"})
    assert "D major" in text
