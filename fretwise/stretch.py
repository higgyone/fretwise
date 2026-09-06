"""Stage 7 - pitch-preserving speed changes.

Pre-renders the clip at slower speeds so a practice view can switch between
them instantly rather than stretching audio in the browser. Slowing down must
not drop the pitch: the notes have to stay where the fretboard says they are.

Two backends. ffmpeg's atempo filter is the default and needs nothing beyond
the ffmpeg already used for ingest. librosa's phase vocoder is a pure-Python
fallback; it is more prone to a smeared, phasey sound on chords, which is
exactly the material here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .ingest import IngestError, find_ffmpeg

# Half speed is slow enough to follow a fast phrase; three-quarters is the
# usual working speed; full speed is included so a practice view can switch
# between them without special-casing the original file.
DEFAULT_SPEEDS = (0.5, 0.75, 1.0)

# atempo accepts 0.5..2.0 per instance; slower needs the filter chained.
ATEMPO_MIN = 0.5


class StretchError(RuntimeError):
    """Raised when a speed is out of range or the render fails."""


def check_speed(speed: float) -> float:
    if not 0.1 <= speed <= 4.0:
        raise StretchError(f"speed must be between 0.1 and 4.0, got {speed}")
    return float(speed)


def atempo_chain(speed: float) -> str:
    """Express a speed as a chain of atempo filters within their valid range."""
    check_speed(speed)
    factors = []
    remaining = speed
    while remaining < ATEMPO_MIN:
        factors.append(ATEMPO_MIN)
        remaining /= ATEMPO_MIN
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def stretch_with_ffmpeg(source: Path | str, dest: Path | str, speed: float) -> Path:
    """Render ``source`` at ``speed``, keeping pitch, using ffmpeg."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    command = [
        find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(source),
        "-filter:a", atempo_chain(speed), "-c:a", "pcm_s16le", str(dest),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise StretchError(f"ffmpeg failed ({result.returncode}): {result.stderr.strip()}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise StretchError(f"ffmpeg produced no audio at {dest}")
    return dest


def stretch_with_librosa(y: np.ndarray, speed: float) -> np.ndarray:
    """Time-stretch a signal with librosa's phase vocoder, keeping pitch."""
    import librosa

    check_speed(speed)
    if speed == 1.0:
        return y
    return librosa.effects.time_stretch(y, rate=speed).astype(np.float32)


def render_speeds(
    source: Path | str,
    *,
    speeds: tuple[float, ...] = DEFAULT_SPEEDS,
    out_dir: Path | str | None = None,
    backend: str = "ffmpeg",
) -> dict[float, Path]:
    """Render one file per speed. Returns ``{speed: path}``."""
    source = Path(source)
    out_dir = Path(out_dir) if out_dir is not None else source.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[float, Path] = {}
    for speed in speeds:
        dest = out_dir / f"{source.stem}-{speed_label(speed)}.wav"
        if backend == "ffmpeg":
            stretch_with_ffmpeg(source, dest, speed)
        elif backend == "librosa":
            import soundfile as sf

            from .analyze import load_audio

            y, sr = load_audio(source)
            sf.write(str(dest), stretch_with_librosa(y, speed), sr)
        else:
            raise StretchError(f"unknown backend {backend!r}; use ffmpeg or librosa")
        rendered[speed] = dest
    return rendered


def speed_label(speed: float) -> str:
    """A filename-safe label, e.g. 0.75 -> ``75``."""
    return f"{round(speed * 100)}"
