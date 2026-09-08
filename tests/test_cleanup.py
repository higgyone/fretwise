"""Dropping notes that cannot belong to the line around them."""

import pytest

from fretwise.analyze import DetectedNote
from fretwise.cleanup import drop_stray_notes, local_median_midi
from fretwise.notes import midi_to_hz, midi_to_name, name_to_midi


def note(name, time, duration=0.3, confidence=0.5):
    midi = name_to_midi(name)
    return DetectedNote(
        time=time, duration=duration, note=midi_to_name(midi), midi=midi,
        hz=midi_to_hz(midi), cents=0.0, confidence=confidence,
    )


def line(*names, spacing=0.4):
    return [note(name, index * spacing) for index, name in enumerate(names)]


def test_drops_a_note_far_above_the_line():
    notes = line("D3", "F#3", "D3", "C6", "D3", "F#3", "D3")
    assert "C6" not in {n.note for n in drop_stray_notes(notes)}
    assert len(drop_stray_notes(notes)) == len(notes) - 1


def test_keeps_an_octave_doubling():
    """A guitar really does play octaves; only wilder leaps are dropped."""
    notes = line("D3", "F#3", "D3", "D4", "D3", "F#3", "D3")
    assert [n.note for n in drop_stray_notes(notes)] == [n.note for n in notes]


def test_keeps_a_part_that_is_simply_high():
    """A line played high is not a stray - it is the line."""
    notes = line("D5", "F#5", "A5", "D5", "F#5", "A5")
    assert len(drop_stray_notes(notes)) == len(notes)


def test_keeps_everything_when_there_is_no_context():
    """Two notes give nothing to compare against, so neither is judged."""
    notes = line("D3", "C6")
    assert len(drop_stray_notes(notes)) == 2


def test_empty_input():
    assert drop_stray_notes([]) == []


def test_threshold_is_adjustable():
    notes = line("D3", "F#3", "D3", "D4", "D3", "F#3", "D3")
    assert len(drop_stray_notes(notes, max_semitones_above=10)) == len(notes) - 1


def test_local_median_uses_only_nearby_notes():
    notes = [
        note("D3", 0.0), note("D3", 0.4), note("D3", 0.8), note("D3", 1.2),
        note("C6", 30.0),
    ]
    # The distant C6 has no neighbours inside the window, so it is not judged.
    assert local_median_midi(notes, 4) is None
    assert local_median_midi(notes, 1) == pytest.approx(name_to_midi("D3"))


def test_a_note_needs_enough_neighbours_to_be_judged():
    """Fewer than three neighbours is not a line to compare against."""
    notes = [note("D3", 0.0), note("D3", 0.4), note("C6", 0.8)]
    assert local_median_midi(notes, 2) is None
    assert len(drop_stray_notes(notes)) == 3


def test_merges_a_held_note_split_in_two():
    from fretwise.cleanup import merge_held_notes

    pieces = [note("D3", 0.0, duration=0.4), note("D3", 0.42, duration=0.4)]
    merged = merge_held_notes(pieces)
    assert len(merged) == 1
    assert merged[0].time == 0.0
    assert merged[0].duration == pytest.approx(0.82)


def test_keeps_a_note_played_again():
    """A clear gap means the string was picked again, not held."""
    from fretwise.cleanup import merge_held_notes

    pieces = [note("D3", 0.0, duration=0.4), note("D3", 0.9, duration=0.4)]
    assert len(merge_held_notes(pieces)) == 2


def test_merging_keeps_different_pitches_apart():
    from fretwise.cleanup import merge_held_notes

    pieces = [note("D3", 0.0, duration=0.4), note("F#3", 0.41, duration=0.4)]
    assert len(merge_held_notes(pieces)) == 2


def test_merging_joins_a_long_run_of_fragments():
    from fretwise.cleanup import merge_held_notes

    pieces = [note("D3", i * 0.41, duration=0.4) for i in range(5)]
    merged = merge_held_notes(pieces)
    assert len(merged) == 1
    assert merged[0].duration == pytest.approx(4 * 0.41 + 0.4)


def test_merging_handles_overlapping_pieces():
    from fretwise.cleanup import merge_held_notes

    pieces = [note("D3", 0.0, duration=0.5), note("D3", 0.3, duration=0.5)]
    merged = merge_held_notes(pieces)
    assert len(merged) == 1
    assert merged[0].duration == pytest.approx(0.8)


def test_merging_keeps_the_strongest_confidence():
    from fretwise.cleanup import merge_held_notes

    pieces = [
        note("D3", 0.0, duration=0.4, confidence=0.7),
        note("D3", 0.42, duration=0.4, confidence=0.3),
    ]
    assert merge_held_notes(pieces)[0].confidence == pytest.approx(0.7)


def test_merging_does_not_mutate_the_input():
    from fretwise.cleanup import merge_held_notes

    pieces = [note("D3", 0.0, duration=0.4), note("D3", 0.42, duration=0.4)]
    merge_held_notes(pieces)
    assert pieces[0].duration == pytest.approx(0.4)


def test_merging_returns_notes_in_time_order():
    from fretwise.cleanup import merge_held_notes

    pieces = [note("F#3", 1.0), note("D3", 0.0), note("A3", 0.5)]
    merged = merge_held_notes(pieces)
    assert [n.time for n in merged] == sorted(n.time for n in merged)


def test_merging_empty_input():
    from fretwise.cleanup import merge_held_notes

    assert merge_held_notes([]) == []


def make_audio(sr=22050, seconds=4.0):
    """A steady tone, so any attack in a test is one the test put there."""
    import numpy as np

    t = np.arange(int(sr * seconds)) / sr
    return (0.2 * np.sin(2 * np.pi * 147 * t)).astype("float32"), sr


def test_attack_ratio_is_about_one_when_nothing_happens():
    from fretwise.cleanup import attack_ratio

    audio, sr = make_audio()
    assert attack_ratio(audio, sr, 2.0) == pytest.approx(1.0, abs=0.05)


def test_attack_ratio_rises_where_the_string_is_struck():
    import numpy as np
    from fretwise.cleanup import attack_ratio

    audio, sr = make_audio()
    audio[int(2.0 * sr):] *= 4  # the signal jumps: a fresh strike
    assert attack_ratio(audio, sr, 2.0) > 3.0


def test_attack_ratio_at_the_edges_is_safe():
    from fretwise.cleanup import attack_ratio

    audio, sr = make_audio()
    assert attack_ratio(audio, sr, 0.0) == pytest.approx(1.0, abs=0.2)
    assert attack_ratio(audio, sr, 99.0) == pytest.approx(1.0, abs=0.2)


def test_a_strum_on_a_ringing_string_stays_a_separate_note():
    """The string sounds on through the strum, so only the audio can tell."""
    from fretwise.cleanup import merge_held_notes

    audio, sr = make_audio()
    audio[int(2.0 * sr):] *= 4
    pieces = [note("D3", 1.4, duration=0.6), note("D3", 2.0, duration=0.6)]

    assert len(merge_held_notes(pieces, audio=audio, sr=sr)) == 2
    # Without the recording there is nothing to go on, and they are joined.
    assert len(merge_held_notes(pieces)) == 1


def test_a_split_with_no_attack_under_it_is_still_joined():
    from fretwise.cleanup import merge_held_notes

    audio, sr = make_audio()
    pieces = [note("D3", 1.4, duration=0.6), note("D3", 2.0, duration=0.6)]
    merged = merge_held_notes(pieces, audio=audio, sr=sr)
    assert len(merged) == 1
    assert merged[0].duration == pytest.approx(1.2)


def test_a_real_gap_is_still_two_notes_whatever_the_audio():
    from fretwise.cleanup import merge_held_notes

    audio, sr = make_audio()
    pieces = [note("D3", 0.5, duration=0.3), note("D3", 2.0, duration=0.3)]
    assert len(merge_held_notes(pieces, audio=audio, sr=sr)) == 2
