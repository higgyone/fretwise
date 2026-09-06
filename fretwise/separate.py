"""Stage 2 — source separation.

Splits a clip into stems with Demucs so the later pitch tracking sees one
instrument instead of a full band mix. This is optional in the pipeline, but
on dense material it is the difference between usable pitch estimates and
noise: pyin is monophonic, and a mix of bass, vocals and guitar gives it
nothing stable to lock onto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Demucs' htdemucs model produces exactly these four stems. Guitar is not one
# of them; lead guitar ends up in "other" along with keys and anything else
# that is not drums, bass or vocals.
STEMS = ("drums", "bass", "other", "vocals")
DEFAULT_STEM = "other"
DEFAULT_MODEL = "htdemucs"


class SeparationError(RuntimeError):
    """Raised when separation is unavailable or fails."""


def is_available() -> bool:
    """Is Demucs usable in this environment?

    Importing demucs pulls in torch, which can fail at load time (a bad DLL,
    a missing runtime) rather than merely being absent — so this catches any
    exception, not just ``ImportError``.
    """
    try:
        import demucs.api  # noqa: F401
    except Exception:
        return False
    return True


def separate_all(
    path: Path | str,
    *,
    model: str = DEFAULT_MODEL,
    out_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Separate ``path`` and write every stem. Returns ``{stem: path}``.

    Demucs computes all stems in a single pass regardless of how many are
    wanted, so keeping them all costs nothing beyond disk and saves a rerun
    when comparing which stem the guitar landed in.
    """
    path = Path(path)
    out_dir = Path(out_dir) if out_dir is not None else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import demucs.api
    except ImportError:
        raise SeparationError(
            "demucs is not installed. `pip install demucs`, or run without --separate."
        ) from None
    except Exception as exc:
        raise SeparationError(f"demucs could not be loaded: {exc}") from exc

    try:
        separator = demucs.api.Separator(model=model, progress=False)
        _origin, stems = separator.separate_audio_file(path)
    except Exception as exc:  # demucs raises a variety of loader/runtime errors
        raise SeparationError(f"demucs failed on {path}: {exc}") from exc

    return {
        name: write_stem(audio, separator.samplerate, out_dir / f"{path.stem}-{name}.wav")
        for name, audio in stems.items()
    }


def separate(
    path: Path | str,
    *,
    stem: str = DEFAULT_STEM,
    model: str = DEFAULT_MODEL,
    out_dir: Path | str | None = None,
) -> Path:
    """Separate ``path`` and write the requested stem beside it as a WAV.

    Returns the path to the stem. Existing output is overwritten.
    """
    if stem not in STEMS:
        raise SeparationError(f"unknown stem {stem!r}; expected one of {', '.join(STEMS)}")

    written = separate_all(path, model=model, out_dir=out_dir)
    if stem not in written:
        raise SeparationError(
            f"model {model!r} produced no {stem!r} stem (got: {', '.join(sorted(written))})"
        )
    return written[stem]


def write_stem(audio, sample_rate: int, destination: Path) -> Path:
    """Write a Demucs stem tensor to a mono WAV."""
    import soundfile as sf

    samples = np.asarray(audio.cpu().numpy() if hasattr(audio, "cpu") else audio)
    if samples.ndim == 2:  # demucs returns (channels, samples)
        samples = samples.mean(axis=0)
    sf.write(str(destination), samples, sample_rate)
    return destination
