"""Note transcription with basic-pitch.

pyin, used by the analyze stage, tracks one pitch at a time. A guitar plays
chords, so on strummed material pyin has to pick a single note out of several
sounding at once -- which caps how much of a part it can ever recover.

basic-pitch is a neural transcriber that returns note events directly, with
starts, ends and overlapping pitches, so a chord comes back as a chord.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .analyze import ANALYSIS_SAMPLE_RATE, DetectedNote, load_audio
from .notes import midi_to_hz, midi_to_name, name_to_midi

# A guitar in standard tuning: open low E up to the 24th fret of the high E.
DEFAULT_MIN_MIDI = name_to_midi("E2")
DEFAULT_MAX_MIDI = name_to_midi("E6")

# basic-pitch's own thresholds. Lower onset/frame values keep more notes.
DEFAULT_ONSET_THRESHOLD = 0.5
DEFAULT_FRAME_THRESHOLD = 0.3
DEFAULT_MIN_NOTE_MS = 80.0


class TranscriptionError(RuntimeError):
    """Raised when basic-pitch is unavailable or fails."""


def is_available() -> bool:
    """Is basic-pitch usable here?

    Importing it pulls in a native inference runtime, which can fail to load
    rather than merely be absent, so this catches any exception.
    """
    try:
        from basic_pitch.inference import predict  # noqa: F401
    except Exception:
        return False
    return True


def transcribe(
    path: Path | str,
    *,
    min_midi: int = DEFAULT_MIN_MIDI,
    max_midi: int = DEFAULT_MAX_MIDI,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    min_note_ms: float = DEFAULT_MIN_NOTE_MS,
) -> list[DetectedNote]:
    """Transcribe a clip to notes, keeping only those in the guitar's range."""
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict
    except ImportError:
        raise TranscriptionError(
            "basic-pitch is not installed. `pip install 'basic-pitch[onnx]'`."
        ) from None
    except Exception as exc:
        raise TranscriptionError(f"basic-pitch could not be loaded: {exc}") from exc

    try:
        _model_output, _midi, events = predict(
            str(path),
            ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_ms,
            minimum_frequency=midi_to_hz(min_midi),
            maximum_frequency=midi_to_hz(max_midi),
        )
    except Exception as exc:
        raise TranscriptionError(f"basic-pitch failed on {path}: {exc}") from exc

    notes = [
        DetectedNote(
            time=float(start),
            duration=float(end - start),
            note=midi_to_name(int(pitch)),
            midi=int(pitch),
            hz=midi_to_hz(int(pitch)),
            # basic-pitch reports a quantised pitch, so there is no offset to
            # measure; pitch bends are reported separately and not used here.
            cents=0.0,
            confidence=float(np.clip(amplitude, 0.0, 1.0)),
        )
        for start, end, pitch, amplitude, _bends in events
        if min_midi <= int(pitch) <= max_midi
    ]
    notes.sort(key=lambda note: (note.time, note.midi))
    return notes


def transcribe_audio(y: np.ndarray, sr: int, **kwargs) -> list[DetectedNote]:
    """Transcribe in-memory audio. basic-pitch reads files, so this writes one."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "audio.wav"
        sf.write(str(path), y, sr)
        return transcribe(path, **kwargs)


def transcribe_file(
    path: Path | str, sample_rate: int = ANALYSIS_SAMPLE_RATE, **kwargs
) -> list[DetectedNote]:
    """Load a clip and transcribe it, resampling first for consistency."""
    y, sr = load_audio(path, sample_rate=sample_rate)
    return transcribe_audio(y, sr, **kwargs)
