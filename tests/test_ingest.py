import subprocess
import wave

import pytest

from fretwise.ingest import IngestError, find_ffmpeg, trim_to_wav


@pytest.fixture(scope="module")
def tone(tmp_path_factory):
    """A 10 second 440 Hz test tone, stereo at 48 kHz, as an m4a."""
    path = tmp_path_factory.mktemp("audio") / "tone.m4a"
    subprocess.run(
        [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:sample_rate=48000:duration=10",
         "-ac", "2", "-c:a", "aac", str(path)],
        check=True,
    )
    return path


def read_wav(path):
    with wave.open(str(path)) as handle:
        return handle.getnchannels(), handle.getframerate(), handle.getnframes()


def test_trim_produces_mono_wav_of_requested_length(tone, tmp_path):
    dest = trim_to_wav(tone, tmp_path / "clip.wav", start=2.0, end=5.0, sample_rate=22050)
    channels, rate, frames = read_wav(dest)
    assert (channels, rate) == (1, 22050)
    assert frames / rate == pytest.approx(3.0, abs=0.05)


def test_trim_without_end_runs_to_the_end_of_source(tone, tmp_path):
    dest = trim_to_wav(tone, tmp_path / "clip.wav", start=8.0)
    _, rate, frames = read_wav(dest)
    assert frames / rate == pytest.approx(2.0, abs=0.05)


def test_trim_rejects_end_before_start(tone, tmp_path):
    with pytest.raises(IngestError, match="must be after"):
        trim_to_wav(tone, tmp_path / "clip.wav", start=5.0, end=2.0)


def test_trim_reports_ffmpeg_failure(tmp_path):
    bogus = tmp_path / "not-audio.m4a"
    bogus.write_text("this is not audio")
    with pytest.raises(IngestError, match="ffmpeg failed"):
        trim_to_wav(bogus, tmp_path / "clip.wav")
