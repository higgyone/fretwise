"""Stage 2 tests. The demucs run itself is heavy, so it is opt-in."""

import numpy as np
import pytest
import soundfile as sf

from fretwise import separate as sep


def test_rejects_an_unknown_stem(tmp_path):
    with pytest.raises(sep.SeparationError, match="unknown stem"):
        sep.separate(tmp_path / "clip.wav", stem="banjo")


def test_the_default_model_has_a_guitar_stem():
    """The whole point of the 6-source model: a four-source one has no guitar."""
    assert "guitar" in sep.MODEL_STEMS[sep.DEFAULT_MODEL]
    assert sep.DEFAULT_STEM == "guitar"
    assert "guitar" not in sep.STEMS_4


def test_reports_a_clear_error_when_demucs_is_missing(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_demucs(name, *args, **kwargs):
        if name.startswith("demucs"):
            raise ImportError("no demucs")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_demucs)
    assert sep.is_available() is False
    with pytest.raises(sep.SeparationError, match="not installed"):
        sep.separate(tmp_path / "clip.wav")


def test_write_stem_downmixes_stereo_to_mono(tmp_path):
    stereo = np.stack([np.ones(100), -np.ones(100) * 0.5])  # (channels, samples)
    dest = sep.write_stem(stereo, 44100, tmp_path / "stem.wav")
    audio, rate = sf.read(dest)
    assert rate == 44100
    assert audio.ndim == 1
    assert audio == pytest.approx(np.full(100, 0.25), abs=1e-4)


@pytest.mark.skipif(not sep.is_available(), reason="demucs not installed")
def test_separates_a_real_mix(tmp_path):
    """A tone plus noise bursts should split into distinct stems."""
    sr = 44100
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    tone = 0.4 * np.sin(2 * np.pi * 220 * t)
    clicks = np.zeros_like(t)
    for onset in range(0, 3 * sr, sr // 2):
        clicks[onset : onset + 200] = np.random.default_rng(0).normal(0, 0.5, 200)
    clip = tmp_path / "clip.wav"
    sf.write(clip, (tone + clicks).astype(np.float32), sr)

    stem = sep.separate(clip, stem="other", model="htdemucs", out_dir=tmp_path)
    assert stem.exists() and stem.name == "clip-other.wav"
    audio, rate = sf.read(stem)
    assert audio.ndim == 1 and len(audio) > 0


@pytest.mark.skipif(not sep.is_available(), reason="demucs not installed")
def test_separate_all_writes_every_stem(tmp_path):
    sr = 44100
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    sf.write(tmp_path / "clip.wav", (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)

    written = sep.separate_all(tmp_path / "clip.wav", model="htdemucs", out_dir=tmp_path)
    assert set(written) == set(sep.MODEL_STEMS["htdemucs"])
    assert all(path.exists() for path in written.values())


def test_energy_share_sums_to_one(tmp_path):
    from fretwise.separate import energy_share

    loud = tmp_path / "loud.wav"
    quiet = tmp_path / "quiet.wav"
    sf.write(loud, np.full(1000, 0.8, dtype="float32"), 22050)
    sf.write(quiet, np.full(1000, 0.2, dtype="float32"), 22050)

    share = energy_share({"loud": loud, "quiet": quiet})
    assert sum(share.values()) == pytest.approx(1.0)
    assert share["loud"] == pytest.approx(0.8)


def test_energy_share_is_ordered_loudest_first(tmp_path):
    from fretwise.separate import energy_share

    for name, level in (("a", 0.1), ("b", 0.9), ("c", 0.5)):
        sf.write(tmp_path / f"{name}.wav", np.full(500, level, dtype="float32"), 22050)
    share = energy_share({n: tmp_path / f"{n}.wav" for n in "abc"})
    assert list(share) == ["b", "c", "a"]


def test_energy_share_handles_a_silent_stem(tmp_path):
    """A stem separation left empty must not divide by zero."""
    from fretwise.separate import energy_share

    sf.write(tmp_path / "silent.wav", np.zeros(500, dtype="float32"), 22050)
    sf.write(tmp_path / "sound.wav", np.full(500, 0.5, dtype="float32"), 22050)
    share = energy_share({n: tmp_path / f"{n}.wav" for n in ("silent", "sound")})
    assert share["silent"] == 0.0
    assert share["sound"] == pytest.approx(1.0)
