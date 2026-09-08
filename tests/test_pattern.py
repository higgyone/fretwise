"""Averaging a repeating figure to the notes most repetitions agree on."""

import random

import pytest

from fretwise import pattern
from fretwise.analyze import DetectedNote
from fretwise.notes import midi_to_hz, midi_to_name, name_to_midi


def note(name, time, *, duration=0.2, confidence=0.6):
    midi = name_to_midi(name)
    return DetectedNote(
        time=time, duration=duration, note=midi_to_name(midi), midi=midi,
        hz=midi_to_hz(midi), cents=0.0, confidence=confidence,
    )


def looping(figure, period, times, *, start=0.0):
    """Play ``figure`` -- (offset, name) pairs -- ``times`` times over."""
    out = []
    for turn in range(times):
        for offset, name in figure:
            out.append(note(name, start + turn * period + offset))
    return out


FIGURE = [(0.0, "D3"), (0.5, "F#3"), (1.0, "A3"), (1.5, "D4")]


def test_the_period_of_a_looping_figure_is_found():
    found, strength = pattern.find_period(looping(FIGURE, 2.0, 12))
    assert found == pytest.approx(2.0, abs=0.1)
    assert strength > 0.3


def test_random_playing_has_no_strong_period():
    rng = random.Random(0)
    scattered = [note("D3", rng.uniform(0, 30)) for _ in range(60)]
    _found, strength = pattern.find_period(scattered)
    assert strength < 0.3


def test_no_notes_has_no_period():
    assert pattern.find_period([]) == (0.0, 0.0)


def test_folding_stacks_the_repetitions():
    counts = pattern.fold(looping(FIGURE, 2.0, 10), 2.0, 0.0, divisions=16)
    # Four distinct pitches, each landing in one slot, ten times over.
    assert len(counts) == 4
    assert all(count == 10 for count in counts.values())


def test_sharpness_is_higher_for_a_loop_than_for_noise():
    rng = random.Random(1)
    scattered = [note("D3", rng.uniform(0, 24)) for _ in range(48)]
    looped = looping(FIGURE, 2.0, 12)
    assert pattern.sharpness(looped, 2.0, 0.0) > pattern.sharpness(scattered, 2.0, 0.0)


def test_the_phase_lines_the_repetitions_up():
    """Where the figure "starts" is arbitrary; stacking cleanly is not.

    An evenly spaced figure has several phases that fold it just as tightly,
    so the test is that the chosen one recovers the figure, not its value.
    """
    played = looping(FIGURE, 2.0, 12, start=0.7)
    phase = pattern.best_phase(played, 2.0)
    counts = pattern.fold(played, 2.0, phase)
    assert len(counts) == 4  # four pitches, each in one slot
    assert all(count == 12 for count in counts.values())


def test_consensus_keeps_what_repeats():
    found, summary = pattern.consensus(looping(FIGURE, 2.0, 12), period=2.0)
    assert [n.note for n in found] == ["D3", "F#3", "A3", "D4"]
    assert all(n.confidence > 0.9 for n in found)
    assert summary["repetitions"] >= 11


def test_consensus_drops_a_note_played_only_once():
    played = looping(FIGURE, 2.0, 12)
    played.append(note("C5", 4.25))  # a single wrong note in one repetition
    found, _summary = pattern.consensus(played, period=2.0, min_share=0.4)
    assert "C5" not in {n.note for n in found}


def test_a_note_in_most_repetitions_survives():
    """The point of averaging: what is mostly there is kept, not discarded."""
    played = looping(FIGURE, 2.0, 12)
    for turn in range(9):  # present in 9 of 12
        played.append(note("B3", turn * 2.0 + 1.75))
    found, _summary = pattern.consensus(played, period=2.0, min_share=0.5)
    assert "B3" in {n.note for n in found}


def test_the_share_is_reported_as_confidence():
    played = looping(FIGURE, 2.0, 12)
    for turn in range(6):
        played.append(note("B3", turn * 2.0 + 1.75))
    found, _summary = pattern.consensus(played, period=2.0, min_share=0.3)
    partial = next(n for n in found if n.note == "B3")
    assert partial.confidence == pytest.approx(0.5, abs=0.1)


def test_the_figure_comes_back_as_one_pass_not_many():
    found, summary = pattern.consensus(looping(FIGURE, 2.0, 12), period=2.0)
    assert max(n.time for n in found) < summary["period"]


def test_consensus_with_no_notes():
    found, summary = pattern.consensus([])
    assert found == [] and summary["repetitions"] == 0


def test_consensus_survives_notes_that_do_not_repeat():
    rng = random.Random(2)
    scattered = [note("D3", rng.uniform(0, 30)) for _ in range(40)]
    found, _summary = pattern.consensus(scattered, min_share=0.8)
    assert found == []


def test_too_few_repetitions_to_average():
    """Two passes agreeing says nothing: any coincidence is unanimous."""
    found, summary = pattern.consensus(looping(FIGURE, 2.0, 2), period=2.0)
    assert found == []
    assert summary["repetitions"] < pattern.MIN_REPETITIONS


def test_positions_are_cleared_so_they_are_mapped_afresh():
    """A folded note is at a new time, so an old string choice is meaningless."""
    played = looping(FIGURE, 2.0, 12)
    for n in played:
        n.chosen = {"string": 6, "fret": 12}
    found, _summary = pattern.consensus(played, period=2.0)
    assert all(n.chosen is None for n in found)


def test_applying_restores_a_note_missed_in_one_repetition():
    """The whole point: what the other repetitions saw fills the gap."""
    played = looping(FIGURE, 2.0, 8)
    played = [n for n in played if not (n.note == "A3" and 6.0 <= n.time < 8.0)]
    assert sum(1 for n in played if n.note == "A3") == 7

    figure, summary = pattern.consensus(played, period=2.0)
    rebuilt, _stats = pattern.apply_to_timeline(played, figure, summary)
    assert sum(1 for n in rebuilt if n.note == "A3") == 8


def test_applying_drops_a_note_played_only_once():
    played = looping(FIGURE, 2.0, 8)
    played.append(note("C5", 5.1))
    figure, summary = pattern.consensus(played, period=2.0)
    rebuilt, _stats = pattern.apply_to_timeline(played, figure, summary)
    assert "C5" not in {n.note for n in rebuilt}


def test_applying_leaves_notes_outside_the_range_alone():
    played = looping(FIGURE, 2.0, 8)
    elsewhere = note("G5", 40.0)
    figure, summary = pattern.consensus(played, period=2.0)
    rebuilt, _stats = pattern.apply_to_timeline(
        played + [elsewhere], figure, summary, start=0.0, end=16.0
    )
    assert "G5" in {n.note for n in rebuilt}


def test_applying_nothing_changes_nothing():
    played = looping(FIGURE, 2.0, 8)
    rebuilt, stats = pattern.apply_to_timeline(played, [], {"period": 0.0})
    assert len(rebuilt) == len(played)
    assert stats["replaced"] == 0


def test_applying_reports_what_it_did():
    played = looping(FIGURE, 2.0, 8)
    figure, summary = pattern.consensus(played, period=2.0)
    _rebuilt, stats = pattern.apply_to_timeline(played, figure, summary)
    assert stats["replaced"] == len(played)
    assert stats["rebuilt"] > 0


def test_applied_notes_are_positioned_afresh():
    played = looping(FIGURE, 2.0, 8)
    figure, summary = pattern.consensus(played, period=2.0)
    rebuilt, _stats = pattern.apply_to_timeline(played, figure, summary)
    assert all(n.chosen is None for n in rebuilt)
