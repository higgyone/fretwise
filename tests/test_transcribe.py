"""Transcription with basic-pitch."""

import sys
from pathlib import Path

import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from fretwise import transcribe as tr
from fretwise.notes import name_to_midi
from test_analyze import SR, melody

needs_basic_pitch = pytest.mark.skipif(
    not tr.is_available(), reason="basic-pitch not installed"
)


@pytest.fixture(scope="module")
def phrase(tmp_path_factory):
    path = tmp_path_factory.mktemp("audio") / "phrase.wav"
    sf.write(path, melody(["E3", "A3", "D4", "G4"], 0.6, 0.15), SR)
    return path


def test_reports_a_clear_error_when_missing(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_basic_pitch(name, *args, **kwargs):
        if name.startswith("basic_pitch"):
            raise ImportError("no basic_pitch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_basic_pitch)
    assert tr.is_available() is False
    with pytest.raises(tr.TranscriptionError, match="not installed"):
        tr.transcribe(tmp_path / "clip.wav")


def test_the_guitar_range_defaults_are_standard_tuning():
    assert tr.DEFAULT_MIN_MIDI == name_to_midi("E2")
    assert tr.DEFAULT_MAX_MIDI == name_to_midi("E6")


@needs_basic_pitch
def test_finds_the_notes_of_a_phrase(phrase):
    notes = tr.transcribe(phrase)
    found = {n.note for n in notes}
    assert {"E3", "A3", "D4", "G4"} <= found, f"missed some of the phrase: {found}"


@needs_basic_pitch
def test_notes_come_back_in_time_order(phrase):
    notes = tr.transcribe(phrase)
    assert [n.time for n in notes] == sorted(n.time for n in notes)


@needs_basic_pitch
def test_notes_carry_duration_and_confidence(phrase):
    notes = tr.transcribe(phrase)
    assert all(n.duration > 0 for n in notes)
    assert all(0.0 <= n.confidence <= 1.0 for n in notes)


@needs_basic_pitch
def test_pitches_outside_the_requested_range_are_dropped(phrase):
    """Asking for a narrow range must exclude notes outside it."""
    notes = tr.transcribe(
        phrase, min_midi=name_to_midi("D4"), max_midi=name_to_midi("G4")
    )
    assert notes, "the narrow range should still contain D4 and G4"
    assert all(name_to_midi("D4") <= n.midi <= name_to_midi("G4") for n in notes)


@needs_basic_pitch
def test_transcribes_overlapping_notes(tmp_path):
    """A chord must come back as several notes, which is the point of this engine."""
    import numpy as np
    from test_analyze import pluck
    from fretwise.notes import midi_to_hz

    chord = sum(
        pluck(midi_to_hz(name_to_midi(n)), 2.0, SR, decay=1.2)
        for n in ("D3", "A3", "D4", "F#4")
    )
    path = tmp_path / "chord.wav"
    sf.write(path, (chord / np.abs(chord).max() * 0.8).astype("float32"), SR)

    notes = tr.transcribe(path)
    # At least two notes of the chord sound at the same moment.
    overlapping = [
        (a, b) for a in notes for b in notes
        if a is not b and a.time < b.time + b.duration and b.time < a.time + a.duration
    ]
    assert overlapping, f"no overlapping notes found in a chord: {[n.note for n in notes]}"


@needs_basic_pitch
def test_transcribe_audio_accepts_an_array(phrase):
    import soundfile

    y, sr = soundfile.read(phrase)
    assert tr.transcribe_audio(y, sr)
