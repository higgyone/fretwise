# fretwise

Extract lead guitar notes from a YouTube clip, then practice them with an
interactive fretboard view — slow it down, loop a phrase, and see exactly
where to put your fingers.

## Why

Ear training is hard. This tool takes a YouTube guitar sample, figures out
what notes are being played and when, and turns that into a visual,
playable-along-with practice tool.

## Pipeline

1. **Ingest** — `yt-dlp` downloads audio from a YouTube URL; `ffmpeg` trims
   to a start/end timestamp range.
2. **Separate** *(optional)* — Demucs splits the mix into stems so lead
   guitar can be isolated from drums/vocals/rhythm before pitch tracking.
3. **Onset detection** — `librosa.onset.onset_detect` finds where each note
   starts.
4. **Pitch tracking** — `librosa.pyin` (or `crepe` for noisy/distorted
   signals) estimates the fundamental frequency per note segment.
5. **Note mapping** — Hz is snapped to the nearest semitone and converted to
   a note name (e.g. `E4`).
6. **Fretboard mapping** — each note is mapped to all valid `(string, fret)`
   positions in standard tuning, with a default position chosen to minimize
   hand movement from the previous note.
7. **Audio rendering** — `pyrubberband` / `librosa.effects.time_stretch`
   pre-render pitch-preserving slowed versions of the clip (e.g. 50%, 75%,
   100% speed).
8. **Practice view** — a static HTML/JS page renders an SVG fretboard with a
   playhead synced to audio, a speed selector, and loop-region markers.

## Output format

`notes.json` — one entry per detected note:

```json
{
  "time": 12.34,
  "duration": 0.42,
  "note": "E4",
  "confidence": 0.91,
  "options": [{"string": 1, "fret": 0}, {"string": 2, "fret": 5}],
  "chosen": {"string": 1, "fret": 0}
}
```

## Config knobs

- `fmin` / `fmax` — pitch search range (default guitar range: E2–E6)
- `separate` — on/off for Demucs source separation
- `onset_sensitivity` — tune false-positive/false-negative onset tradeoff
- `output_format` — `json`, `csv`, or ASCII tab

## Stack

- **Python**: `yt-dlp`, `librosa`, `demucs`, `pyrubberband`, `numpy`
- **Web**: single static HTML file, embedded JS, SVG fretboard — no build
  tooling required. Deployable as-is via GitHub Pages, so practice sessions
  work from a phone browser once pushed.

## Status

- Stage 1 (ingest) — built: `fretwise ingest <url> --start 1:23 --end 1:41`
- Stage 2 (separate) — built: `fretwise separate --all`
- Stages 3-5 (onsets, pitch, note names) — built: `fretwise analyze --separate`
- Stages 6, 7, 8 — not yet built

Run end to end on a real full-band track. Two things to know:

**Separation is not optional on a band mix.** `pyin` tracks one pitch at a
time, so on the raw mix every segment came back below the confidence floor.
Demucs lifts it into usable range. `htdemucs` has no guitar stem — lead
guitar lands in `other`, alongside keys and anything else pitched.

**Recall is low.** A 150s clip yielded 21 notes where the real part has many
more. Precision looks reasonable (86% of detected pitches fell inside a
single major key, against ~58% for random pitches, and a drum stem correctly
yields nothing), but the detector is missing most of what is played. That is
the next thing worth working on.

## Roadmap / stretch goals

- Re-listen to a flagged/low-confidence segment in isolation
- ASCII tab-style export
- Confidence-based visual highlighting of unreliable note guesses
