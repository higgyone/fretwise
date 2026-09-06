"""Mapping notes onto the fretboard."""

import pytest

from fretwise.analyze import DetectedNote
from fretwise.fretboard import (
    STANDARD_TUNING,
    Position,
    choose,
    map_notes,
    positions,
    reach_cost,
    tuning_midi,
)
from fretwise.notes import midi_to_hz, midi_to_name, name_to_midi


def note(name, time, duration=0.3):
    midi = name_to_midi(name)
    return DetectedNote(
        time=time, duration=duration, note=midi_to_name(midi), midi=midi,
        hz=midi_to_hz(midi), cents=0.0, confidence=0.6,
    )


def places(name):
    return [(p.string, p.fret) for p in positions(name_to_midi(name))]


def test_open_strings_are_where_they_should_be():
    for string, name in enumerate(STANDARD_TUNING, start=1):
        assert (string, 0) in places(name)


def test_a_pitch_is_found_everywhere_it_can_be_played():
    # E4: open high E, 5th fret of B, 9th of G, 14th of D, 19th of A.
    assert places("E4") == [(1, 0), (2, 5), (3, 9), (4, 14), (5, 19)]


def test_the_lowest_note_has_only_one_place():
    assert places("E2") == [(6, 0)]


def test_a_pitch_below_the_instrument_has_none():
    assert places("E1") == []


def test_a_pitch_above_the_last_fret_has_none():
    assert positions(name_to_midi("E6"), max_fret=5) == []


def test_tuning_is_in_descending_pitch_order():
    """String 1 is the thinnest and highest; string 6 the lowest."""
    midis = tuning_midi()
    assert midis == sorted(midis, reverse=True)


def test_reach_cost_prefers_staying_put():
    near = Position(string=2, fret=5)
    far = Position(string=5, fret=19)
    assert reach_cost(near, anchor=5) < reach_cost(far, anchor=5)
    assert reach_cost(far, anchor=18) < reach_cost(near, anchor=18)


def test_an_open_string_costs_no_movement():
    assert reach_cost(Position(string=1, fret=0), anchor=15)[0] == 0


def test_a_chord_does_not_put_two_notes_on_one_string():
    """E4 and B3 can both be played on string 2; they must not both be."""
    picked = choose([name_to_midi("E4"), name_to_midi("B3")])
    strings = [p.string for p in picked.values() if p]
    assert len(strings) == len(set(strings))


def test_a_chord_places_the_most_constrained_note_first():
    """E2 has only one home; a freer note must not take string 6 first."""
    picked = choose([name_to_midi("E2"), name_to_midi("E3")])
    assert picked[name_to_midi("E2")] == Position(string=6, fret=0)
    assert picked[name_to_midi("E3")] is not None


def test_a_note_with_nowhere_left_to_go_is_unassigned():
    picked = choose([name_to_midi("E2")], taken={6})
    assert picked[name_to_midi("E2")] is None


def test_map_notes_fills_in_options_and_choice():
    mapped = map_notes([note("E4", 0.0)])
    assert mapped[0].options == [
        {"string": 1, "fret": 0}, {"string": 2, "fret": 5}, {"string": 3, "fret": 9},
        {"string": 4, "fret": 14}, {"string": 5, "fret": 19},
    ]
    assert mapped[0].chosen == {"string": 1, "fret": 0}


def test_a_line_stays_in_one_hand_position():
    """Playing up the neck, the next note should be reachable, not at the nut."""
    line = [note("A4", 0.0), note("B4", 0.4), note("C5", 0.8)]
    # Start the hand high by forcing the first note onto a high fret.
    mapped = map_notes(line)
    frets = [n.chosen["fret"] for n in mapped]
    assert max(frets) - min(frets) <= 5, f"hand jumps around: {frets}"


def test_simultaneous_notes_get_different_strings():
    chord = [note("D3", 0.0, duration=1.0), note("A3", 0.0, duration=1.0),
             note("D4", 0.0, duration=1.0), note("F#4", 0.0, duration=1.0)]
    mapped = map_notes(chord)
    strings = [n.chosen["string"] for n in mapped if n.chosen]
    assert len(strings) == 4
    assert len(set(strings)) == 4


def test_a_string_frees_up_once_the_note_has_finished():
    """The same string can be reused by a later, non-overlapping note."""
    mapped = map_notes([note("E2", 0.0, duration=0.2), note("E2", 1.0, duration=0.2)])
    assert all(n.chosen == {"string": 6, "fret": 0} for n in mapped)


def test_map_notes_returns_notes_in_time_order():
    mapped = map_notes([note("D3", 1.0), note("A3", 0.0)])
    assert [n.time for n in mapped] == [0.0, 1.0]


def test_map_notes_with_no_notes():
    assert map_notes([]) == []


def test_a_note_does_not_take_the_string_a_coming_note_needs():
    """D3 has three homes; G2 has only string 6. D3 must leave it free."""
    line = [note("D3", 0.0, duration=1.5), note("G2", 0.5, duration=0.5)]
    mapped = map_notes(line)
    by_name = {n.note: n.chosen for n in mapped}
    assert by_name["G2"] == {"string": 6, "fret": 3}
    assert by_name["D3"]["string"] != 6


def test_reserving_does_not_block_a_note_with_nowhere_else():
    """If only the reserved string is left, take it rather than give up."""
    line = [note("E2", 0.0, duration=1.0), note("E2", 0.5, duration=0.5)]
    mapped = map_notes(line)
    # The first gets string 6; the second overlaps and genuinely cannot be played.
    assert mapped[0].chosen == {"string": 6, "fret": 0}
    assert mapped[1].chosen is None


def test_lookahead_ignores_notes_starting_after_this_one_ends():
    from fretwise.fretboard import strings_needed_soon

    line = [note("D3", 0.0, duration=0.2), note("G2", 5.0, duration=0.5)]
    assert strings_needed_soon(line, 0) == set()
