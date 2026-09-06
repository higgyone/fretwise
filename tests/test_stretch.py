"""Pitch-preserving speed changes."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from fretwise import stretch
from test_analyze import SR, melody

SPEEDS = [0.5, 0.75, 1.0]


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("audio") / "clip.wav"
    sf.write(path, melody(["E3", "A3", "D4", "G4"], 0.5, 0.1), SR)
    return path


def duration(path):
    info = sf.info(str(path))
    return info.frames / info.samplerate


def dominant_hz(audio, sr):
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return float(np.fft.rfftfreq(len(audio), 1 / sr)[spectrum.argmax()])


@pytest.mark.parametrize("speed", SPEEDS)
def test_duration_scales_with_speed(clip, tmp_path, speed):
    dest = stretch.stretch_with_ffmpeg(clip, tmp_path / f"out{speed}.wav", speed)
    assert duration(dest) == pytest.approx(duration(clip) / speed, rel=0.02)


@pytest.mark.parametrize("speed", SPEEDS)
def test_pitch_is_preserved(clip, tmp_path, speed):
    """The whole point: slower must not mean lower."""
    dest = stretch.stretch_with_ffmpeg(clip, tmp_path / f"pitch{speed}.wav", speed)
    original, sr = sf.read(clip)
    slowed, _ = sf.read(dest)

    # Compare the first note of each: same pitch, however long it now lasts.
    before = dominant_hz(original[: int(0.4 * sr)], sr)
    after = dominant_hz(slowed[: int(0.4 / speed * sr)], sr)
    assert after == pytest.approx(before, rel=0.03)


def test_atempo_chains_for_very_slow_speeds():
    """atempo only accepts 0.5 and above, so slower speeds must chain."""
    assert stretch.atempo_chain(0.5).count("atempo") == 1
    assert stretch.atempo_chain(0.25).count("atempo") == 2
    factors = [float(part.split("=")[1]) for part in stretch.atempo_chain(0.25).split(",")]
    assert np.prod(factors) == pytest.approx(0.25)


@pytest.mark.parametrize("speed", [0.3, 0.25, 0.1])
def test_chained_speeds_still_scale_correctly(clip, tmp_path, speed):
    dest = stretch.stretch_with_ffmpeg(clip, tmp_path / f"slow{speed}.wav", speed)
    assert duration(dest) == pytest.approx(duration(clip) / speed, rel=0.05)


@pytest.mark.parametrize("speed", [0, -1, 5.0, 100])
def test_impossible_speeds_are_rejected(speed):
    with pytest.raises(stretch.StretchError, match="speed must be"):
        stretch.check_speed(speed)


def test_render_speeds_writes_one_file_each(clip, tmp_path):
    rendered = stretch.render_speeds(clip, speeds=(0.5, 1.0), out_dir=tmp_path)
    assert set(rendered) == {0.5, 1.0}
    assert all(path.exists() for path in rendered.values())
    assert rendered[0.5].name == "clip-50.wav"
    assert rendered[1.0].name == "clip-100.wav"


def test_render_speeds_rejects_an_unknown_backend(clip, tmp_path):
    with pytest.raises(stretch.StretchError, match="unknown backend"):
        stretch.render_speeds(clip, speeds=(1.0,), out_dir=tmp_path, backend="magic")


def test_speed_labels():
    assert stretch.speed_label(0.5) == "50"
    assert stretch.speed_label(0.75) == "75"
    assert stretch.speed_label(1.0) == "100"


def test_librosa_backend_also_preserves_pitch(clip, tmp_path):
    from fretwise.analyze import load_audio

    y, sr = load_audio(clip)
    slowed = stretch.stretch_with_librosa(y, 0.5)
    assert len(slowed) == pytest.approx(len(y) * 2, rel=0.05)
    assert dominant_hz(slowed[: int(0.8 * sr)], sr) == pytest.approx(
        dominant_hz(y[: int(0.4 * sr)], sr), rel=0.03
    )


def test_librosa_backend_leaves_full_speed_untouched():
    y = np.ones(100, dtype=np.float32)
    assert stretch.stretch_with_librosa(y, 1.0) is y
