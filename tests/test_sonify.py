"""Rendering the detected notes back to audio."""

import numpy as np
import pytest

from fretwise import sonify
from fretwise.analyze import DetectedNote
from fretwise.notes import midi_to_hz, name_to_midi

SR = 22050


def note(name, time, duration=0.4):
    midi = name_to_midi(name)
    return DetectedNote(
        time=time, duration=duration, note=name, midi=midi,
        hz=midi_to_hz(midi), cents=0.0, confidence=0.9,
    )


def dominant_hz(audio, sr):
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return float(np.fft.rfftfreq(len(audio), 1 / sr)[spectrum.argmax()])


def test_synth_note_has_the_requested_pitch_and_length():
    tone = sonify.synth_note(440.0, 0.5, SR)
    assert len(tone) == pytest.approx(SR * 0.5, abs=2)
    assert dominant_hz(tone, SR) == pytest.approx(440.0, rel=0.02)


def test_synth_note_ends_in_silence():
    """A hard cut would click; the tail must ramp down."""
    tone = sonify.synth_note(440.0, 0.5, SR)
    assert abs(tone[-1]) < 1e-3


def test_render_places_each_note_at_its_own_time():
    rendered = sonify.render([note("A4", 0.0), note("A4", 1.0)], SR, duration=2.0)
    assert len(rendered) == SR * 2
    # Loud where the notes are, quiet in the gap between them.
    def energy(start, end):
        return float(np.abs(rendered[int(start * SR) : int(end * SR)]).mean())
    assert energy(0.0, 0.3) > 10 * energy(0.65, 0.95)
    assert energy(1.0, 1.3) > 10 * energy(0.65, 0.95)


def test_render_reproduces_the_pitch():
    rendered = sonify.render([note("E4", 0.0, duration=1.0)], SR, duration=1.0)
    assert dominant_hz(rendered[: SR // 2], SR) == pytest.approx(
        midi_to_hz(name_to_midi("E4")), rel=0.02
    )


def test_render_with_no_notes_is_silent():
    assert not sonify.render([], SR, duration=1.0).any()


def test_render_does_not_run_past_the_requested_duration():
    """A note whose tail exceeds the clip is truncated, not an index error."""
    rendered = sonify.render([note("A4", 0.9, duration=2.0)], SR, duration=1.0)
    assert len(rendered) == SR


def test_normalize_leaves_silence_alone():
    silence = np.zeros(100, dtype=np.float32)
    assert not sonify.normalize(silence).any()


def test_normalize_scales_to_the_headroom():
    assert np.abs(sonify.normalize(np.array([0.1, -0.05]))).max() == pytest.approx(0.95)


def test_mix_contains_both_parts():
    clip = np.sin(2 * np.pi * 100 * np.arange(SR) / SR).astype(np.float32)
    rendered = sonify.render([note("A4", 0.0, duration=1.0)], SR, duration=1.0)
    mixed = sonify.mix(clip, rendered)
    spectrum = np.abs(np.fft.rfft(mixed))
    freqs = np.fft.rfftfreq(len(mixed), 1 / SR)

    def level(hz):
        return spectrum[np.argmin(np.abs(freqs - hz))]

    assert level(100) > 10 * level(700)  # the original tone
    assert level(440) > 10 * level(700)  # the rendered note


def test_mix_handles_differing_lengths():
    mixed = sonify.mix(np.ones(SR, dtype=np.float32), np.ones(SR // 2, dtype=np.float32))
    assert len(mixed) == SR


def test_write_round_trips(tmp_path):
    import soundfile as sf

    audio = sonify.render([note("A4", 0.0)], SR, duration=1.0)
    path = sonify.write(audio, SR, tmp_path / "out.wav")
    back, rate = sf.read(path)
    assert rate == SR and len(back) == len(audio)
