"""Re-listening to the notes least worth trusting.

Confidence says which notes the transcription is least sure of, but a number
cannot say whether a note is right. This cuts the recording down to just those
moments, with the transcription played over them exactly as `sonify` does --
the same thing to listen for, without scrubbing through the whole clip to
find the eight places worth checking.

The recording is normalised segment by segment, because low confidence and
low volume tend to arrive together and the passage in question is often the
quietest thing on the clip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sonify import normalize, synth_note

# Heard either side of the note, for context: a note alone is hard to place.
PAD = 0.6
# Between the recording and the note it was read as.
GAP = 0.2
# After each pair, before the next.
SEPARATION = 0.7
# A reference tone longer than this outstays its welcome.
MAX_TONE = 1.2
# A doubtful note often lasts a tenth of a second, which is too brief to hear
# as a pitch at all. The reference is stretched to at least this.
MIN_TONE = 0.4
DEFAULT_COUNT = 10


@dataclass
class Item:
    """One doubtful note, and where it sits in the review audio."""

    note: object
    at: float  # seconds into the review file

    @property
    def time(self) -> float:
        return self.note.time


def weakest(notes, *, count: int = DEFAULT_COUNT, below: float | None = None) -> list:
    """The notes least worth trusting, in the order they are played.

    ``below`` takes everything under a confidence; otherwise the ``count``
    weakest. Ordering the result by time keeps the review in step with the
    recording, which makes it far easier to follow.
    """
    if below is not None:
        picked = [n for n in notes if n.confidence < below]
    else:
        picked = sorted(notes, key=lambda n: n.confidence)[: max(count, 0)]
    return sorted(picked, key=lambda n: n.time)


def excerpt(clip: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """A slice of the recording, clamped to what exists."""
    first = max(0, int(start * sr))
    last = min(len(clip), int(end * sr))
    return clip[first:last] if last > first else np.zeros(0, dtype=np.float32)


def build(
    clip: np.ndarray,
    sr: int,
    notes,
    *,
    context: list | None = None,
    pad: float = PAD,
    gap: float = GAP,
    separation: float = SEPARATION,
) -> tuple[np.ndarray, list[Item]]:
    """Cut the recording down to the doubtful moments, notes played over it.

    ``context`` is every note in the piece, so that whatever else is sounding
    in a window is heard too: a note judged in isolation from the rest of the
    playing is no easier to judge than one heard out of time.
    """
    silence = lambda seconds: np.zeros(max(int(seconds * sr), 0), dtype=np.float32)
    context = list(context if context is not None else notes)

    pieces: list[np.ndarray] = []
    items: list[Item] = []
    position = 0.0

    for note in notes:
        start = note.time - pad
        finish = note.time + note.duration + pad
        heard = excerpt(clip, sr, start, finish)
        if not heard.size:
            continue

        # Levelled window by window: a doubtful note is often the quietest
        # passage on the clip and would otherwise be inaudible.
        window = normalize(heard, headroom=0.7).copy()
        for other in context:
            if other.time >= finish or other.time + other.duration <= start:
                continue
            length = min(max(other.duration, MIN_TONE), MAX_TONE)
            tone = synth_note(other.hz, length, sr)
            at = int((other.time - start) * sr)
            first, last = max(at, 0), min(at + len(tone), len(window))
            if last > first:
                window[first:last] += tone[first - at : last - at] * 0.5

        items.append(Item(note=note, at=position))
        for piece in (normalize(window, headroom=0.95), silence(separation)):
            pieces.append(piece)
            position += len(piece) / sr

    if not pieces:
        return np.zeros(0, dtype=np.float32), []
    return np.concatenate(pieces).astype(np.float32), items


def index(items: list[Item]) -> str:
    """A listing to read while the review plays."""
    from .timestamps import format_timestamp

    if not items:
        return "nothing to review\n"

    lines = [
        f"{'in review':>10}  {'in clip':>9}  {'note':<5} {'conf':>5}  where"
    ]
    for item in items:
        note = item.note
        where = (
            f"string {note.chosen['string']} fret {note.chosen['fret']}"
            if getattr(note, "chosen", None)
            else "unplaced"
        )
        lines.append(
            f"{format_timestamp(item.at):>10}  {format_timestamp(note.time):>9}  "
            f"{note.note:<5} {note.confidence:5.2f}  {where}"
        )
    return "\n".join(lines) + "\n"
