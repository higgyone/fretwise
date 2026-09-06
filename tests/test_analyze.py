import numpy as np
import pytest

from fretwise.analyze import analyze, detect_onsets, segment_bounds
from fretwise.notes import midi_to_hz, name_to_midi

SR = 22050


def pluck(hz, duration, sr=SR, decay=6.0, release=0.02):
    """A crude plucked-string tone: a few decaying harmonics.

    The tail is ramped to silence over ``release`` seconds. Without that, the
    hard cut at the end is a broadband click that any onset detector will
    (correctly) hear as a new note.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    envelope = np.exp(-decay * t)
    ramp = int(min(release, duration) * sr)
    if ramp:
        envelope[-ramp:] *= np.linspace(1.0, 0.0, ramp)
    wave = sum(
        (1.0 / (h**1.5)) * np.sin(2 * np.pi * hz * h * t) for h in (1, 2, 3, 4)
    )
    return (wave * envelope).astype(np.float32)


def melody(names, note_duration=0.5, gap=0.1, sr=SR):
    """Concatenate plucked notes separated by short silences."""
    silence = np.zeros(int(sr * gap), dtype=np.float32)
    parts = [silence]
    for name in names:
        parts.append(pluck(midi_to_hz(name_to_midi(name)), note_duration, sr))
        parts.append(silence)
    return np.concatenate(parts)


@pytest.fixture(scope="module")
def phrase():
    names = ["E4", "G4", "B4", "E5"]
    return names, melody(names)


def test_detects_an_onset_at_every_note_start(phrase):
    names, y = phrase
    onsets = detect_onsets(y, SR)
    # Notes start at 0.1s and every 0.6s after (0.5s note + 0.1s gap).
    for index in range(len(names)):
        expected = 0.1 + index * 0.6
        assert min(abs(onsets - expected)) < 0.05, f"no onset near {expected}s"


def test_note_times_advance_and_do_not_overlap(phrase):
    _, y = phrase
    notes = analyze(y, SR)
    for earlier, later in zip(notes, notes[1:]):
        assert earlier.time + earlier.duration <= later.time + 1e-6


def test_silence_yields_no_notes():
    assert analyze(np.zeros(SR, dtype=np.float32), SR) == []


def test_empty_audio_is_safe():
    assert analyze(np.array([], dtype=np.float32), SR) == []


def test_sensitivity_must_be_in_range():
    with pytest.raises(ValueError, match="sensitivity"):
        detect_onsets(np.zeros(SR, dtype=np.float32), SR, sensitivity=1.5)


def test_segment_bounds_runs_last_note_to_the_end():
    assert segment_bounds(np.array([0.0, 1.0, 2.0]), 3.0) == [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
    ]


def test_segment_bounds_drops_onsets_past_the_end():
    assert segment_bounds(np.array([0.5, 9.0]), 2.0) == [(0.5, 2.0)]


def test_to_dict_rounds_floats(phrase):
    _, y = phrase
    entry = analyze(y, SR)[0].to_dict()
    assert set(entry) == {
        "time", "duration", "note", "midi", "hz", "cents", "confidence",
        "degree", "numeral", "options", "chosen",
    }
    assert entry["note"] == "E4"
    assert isinstance(entry["midi"], int)


def test_follows_a_riff_including_accidentals():
    names = ["E4", "G4", "A4", "A#4", "A4", "G4", "E4", "D4"]
    notes = analyze(melody(names, note_duration=0.4, gap=0.08), SR)
    assert [n.note for n in notes] == names


def test_separates_fast_repeated_picking():
    """Six picks of one pitch are six notes, not one long one."""
    notes = analyze(melody(["A4"] * 6, note_duration=0.1, gap=0.02), SR)
    assert [n.note for n in notes] == ["A4"] * 6


def test_spans_the_guitar_range():
    """Every pitch a guitar can produce must be reachable.

    Run with the pitch-transition constraint lifted: this melody leaps an
    octave between adjacent notes, which the default rate deliberately
    suppresses (see test_octave_leaps_are_the_cost_of_the_transition_limit).
    """
    names = ["E2", "A2", "D3", "G3", "B3", "E4", "E5", "E6"]
    notes = analyze(
        melody(names, note_duration=0.4, gap=0.08), SR, max_transition_rate=None
    )
    assert [n.note for n in notes] == names


def test_octave_leaps_are_the_cost_of_the_transition_limit():
    """Capping the pitch transition rate suppresses real octave leaps too.

    This is the trade the default makes: spurious jumps to a harmonic are
    far more common in this material than genuine octave leaps between
    consecutive notes, but the leaps that do occur get flattened.
    """
    leaping = melody(["E4", "E5"], note_duration=0.4, gap=0.08)

    constrained = analyze(leaping, SR)
    assert [n.note for n in constrained] == ["E4", "E4"]  # the leap is flattened

    unconstrained = analyze(leaping, SR, max_transition_rate=None)
    assert [n.note for n in unconstrained] == ["E4", "E5"]  # and recovered


def test_gain_flattening_evens_out_a_fading_signal():
    """A stem that fades should come back at a consistent level."""
    from fretwise.analyze import flatten_gain

    y = melody(["A4"] * 8, note_duration=0.4, gap=0.1)
    fading = (y * np.linspace(1.0, 0.01, len(y))).astype(np.float32)

    flattened = flatten_gain(fading, SR)
    half = len(flattened) // 2

    def level(chunk):
        return float(np.sqrt(np.mean(chunk**2)))

    # Before: the second half is under half the level of the first.
    assert level(fading[half:]) < 0.5 * level(fading[:half])
    # After: the two halves sit within 20% of each other.
    assert level(flattened[half:]) > 0.8 * level(flattened[:half])


def test_gain_flattening_leaves_silence_alone():
    from fretwise.analyze import flatten_gain

    assert not flatten_gain(np.zeros(SR, dtype=np.float32), SR).any()


def test_gain_flattening_handles_audio_shorter_than_the_window():
    from fretwise.analyze import flatten_gain

    short = np.ones(100, dtype=np.float32)
    assert len(flatten_gain(short, SR)) == 100


def test_transition_rate_is_constrained_by_default():
    """The librosa default lets pitch jump to a harmonic; ours does not."""
    from fretwise.analyze import MAX_TRANSITION_RATE

    assert MAX_TRANSITION_RATE < 35.92
