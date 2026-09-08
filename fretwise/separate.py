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

# The four-source models split a mix into these parts only. Guitar is not one
# of them, so on those models a guitar ends up spread across "other" and, for
# lower notes, "bass" -- which is why a guitar tracked through them fades in
# and out as notes cross whatever the model considers bass-like.
STEMS_4 = ("drums", "bass", "other", "vocals")
# htdemucs_6s adds a guitar stem, which is what this tool actually wants.
STEMS_6 = STEMS_4 + ("guitar", "piano")
STEMS = STEMS_6  # accepted on the command line; what a model yields is checked at runtime

MODEL_STEMS = {"htdemucs": STEMS_4, "htdemucs_ft": STEMS_4, "mdx_extra": STEMS_4,
               "htdemucs_6s": STEMS_6}

DEFAULT_MODEL = "htdemucs_6s"
DEFAULT_STEM = "guitar"


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


def energy_share(paths: dict[str, Path]) -> dict[str, float]:
    """What fraction of the total each stem holds, loudest first.

    Separation splits by instrument, and it has no idea what an acoustic
    guitar is: the low end of the neck looks like a bass to it, so a single
    guitar can arrive split across two stems by where it was played. Seeing
    the split is the quickest way to notice that.
    """
    import numpy as np
    import soundfile as sf

    levels: dict[str, float] = {}
    for name, path in paths.items():
        audio, _sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        levels[name] = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

    total = sum(levels.values()) or 1.0
    return dict(sorted(((k, v / total) for k, v in levels.items()),
                       key=lambda kv: -kv[1]))
