"""Stage 8 - the practice view.

Builds a single self-contained HTML page: an SVG fretboard lit up in time
with the audio, a timeline of the whole part, a speed selector backed by the
files stage 7 pre-rendered, and a loop you drag out with a finger or mouse.

The note data is embedded in the page rather than fetched, so the file opens
straight from disk without a web server. Only the audio is referenced by
name, so the page and its .wav files travel together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from statistics import quantiles

from .fretboard import DEFAULT_MAX_FRET, STANDARD_TUNING
from .stretch import DEFAULT_SPEEDS, speed_label

TEMPLATE = Path(__file__).parent / "templates" / "practice.html"

# Practising happens slowly, so a slow speed is the more useful default.
DEFAULT_VIEW_SPEED = 0.75


class ViewError(RuntimeError):
    """Raised when the page cannot be built."""


@dataclass
class Page:
    """A written practice page and what it refers to."""

    path: Path
    notes: int
    speeds: dict[float, str]
    missing_audio: list[str]


def note_payload(note) -> dict:
    """The fields the page actually uses, and no more."""
    chosen = note.chosen or {}
    return {
        "time": round(note.time, 3),
        "duration": round(note.duration, 3),
        "note": note.note,
        # The page labels a dot with the pitch class alone; the octave is
        # implied by where it sits on the neck.
        "pitchClass": note.note.rstrip("-0123456789"),
        "midi": note.midi,
        "confidence": round(note.confidence, 3),
        "degree": note.degree,
        "numeral": note.numeral,
        "string": chosen.get("string"),
        "fret": chosen.get("fret"),
    }


def confidence_range(notes) -> dict:
    """The span of confidences in this clip, for shading notes against.

    Confidence here is how strongly a note was detected, not a probability
    that it is correct, and its range differs from one recording to the next.
    Shading against the clip's own spread says "less strongly detected than
    its neighbours", which is what a player can act on; an absolute threshold
    would imply a certainty the number does not carry.

    The tenth and ninetieth percentiles are used so that one very quiet or
    one very loud note does not flatten everything else.
    """
    values = sorted(note.confidence for note in notes)
    if len(values) < 10:
        low, high = (values[0], values[-1]) if values else (0.0, 1.0)
    else:
        deciles = quantiles(values, n=10)
        low, high = deciles[0], deciles[-1]
    if high - low < 0.05:  # a flat clip: shade everything the same
        low, high = low - 0.025, high + 0.025
    return {"low": round(low, 4), "high": round(high, 4)}


def build_payload(
    notes,
    *,
    key: dict | None,
    speeds: dict[float, str],
    default_speed: float,
    duration: float | None = None,
    max_fret: int = DEFAULT_MAX_FRET,
) -> dict:
    """Everything the page needs, as one JSON-serialisable object."""
    if not speeds:
        raise ViewError("no rendered speeds to play; run `fretwise render` first")
    if default_speed not in speeds:
        default_speed = min(speeds, key=lambda s: abs(s - default_speed))

    if duration is None:
        duration = max((n.time + n.duration for n in notes), default=0.0)

    return {
        "notes": [note_payload(n) for n in notes],
        "key": key,
        "duration": round(max(duration, 0.1), 3),
        "tuning": list(STANDARD_TUNING),
        "confidence": confidence_range(notes),
        "maxFret": max_fret,
        # A list, not an object keyed by speed: JSON writes 1.0 as 1, so a
        # float used as a key does not survive the round trip to JavaScript.
        "speeds": [
            {"speed": speed, "file": speeds[speed]} for speed in sorted(speeds)
        ],
        "defaultSpeed": default_speed,
    }


def render_page(payload: dict, *, title: str) -> str:
    """Fill the template. The data is embedded, so the page needs no server."""
    template = TEMPLATE.read_text(encoding="utf-8")
    # The payload goes inside a <script type="application/json"> block, where
    # only a literal "</script>" could end it early.
    data = json.dumps(payload).replace("</", "<\\/")
    return template.replace("__TITLE__", escape(title)).replace("__DATA__", data)


def escape(text: str) -> str:
    """Minimal HTML escaping for the title, which comes from a video's name."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def find_speeds(
    work_dir: Path, stem: str = "clip", speeds: tuple[float, ...] = DEFAULT_SPEEDS
) -> tuple[dict[float, str], list[str]]:
    """Which pre-rendered speeds exist beside the page, and which are missing."""
    present: dict[float, str] = {}
    missing: list[str] = []
    for speed in speeds:
        name = f"{stem}-{speed_label(speed)}.wav"
        if (work_dir / name).exists():
            present[speed] = name
        else:
            missing.append(name)
    return present, missing


def write_page(
    notes,
    work_dir: Path | str,
    *,
    key: dict | None = None,
    title: str = "Practice",
    speeds: tuple[float, ...] = DEFAULT_SPEEDS,
    default_speed: float = DEFAULT_VIEW_SPEED,
    max_fret: int = DEFAULT_MAX_FRET,
    filename: str = "practice.html",
) -> Page:
    """Write the practice page into ``work_dir``, beside its audio."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    found, missing = find_speeds(work_dir, speeds=speeds)
    payload = build_payload(
        notes, key=key, speeds=found, default_speed=default_speed, max_fret=max_fret
    )

    destination = work_dir / filename
    destination.write_text(render_page(payload, title=title), encoding="utf-8")
    return Page(
        path=destination, notes=len(notes), speeds=found, missing_audio=missing
    )
