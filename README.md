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
4. **Pitch tracking** — `basic-pitch` transcribes note events directly,
   including overlapping ones, so a chord comes back as a chord.
   `librosa.pyin` remains available via `--engine pyin`, but tracks only one
   pitch at a time.
5. **Note mapping** — Hz is snapped to the nearest semitone and converted to
   a note name (e.g. `E4`).
6. **Fretboard mapping** — each note is mapped to all valid `(string, fret)`
   positions in standard tuning, with a default position chosen to minimize
   hand movement from the previous note.
7. **Audio rendering** — ffmpeg's `atempo` filter pre-renders
   pitch-preserving slowed versions of the clip (50%, 75%, 100% by default).
   `librosa.effects.time_stretch` is available as `--backend librosa`.
   `pyrubberband` is not used: it shells out to a `rubberband` binary, where
   ffmpeg is already a dependency of the ingest stage.
8. **Practice view** — `fretwise view` writes a static HTML page: an SVG
   fretboard lit up in time with the audio, a timeline of the whole part,
   a speed selector backed by the stage 7 renders, and a loop you drag out.
   The note data is embedded, so the page opens straight from disk.

## Output format

`notes.json` — the estimated key, then one entry per detected note:

```json
{
  "key": {
    "name": "D major", "tonic": 2, "mode": "major",
    "fit": 0.70, "margin": 0.18, "in_key": 0.78
  },
  "notes": [
    {
      "time": 12.34,
      "duration": 0.42,
      "note": "E4",
      "midi": 64,
      "hz": 329.63,
      "cents": -4.1,
      "confidence": 0.91,
      "degree": "2",
      "numeral": "ii"
    }
  ]
}
```

`degree` is the note's position in the estimated scale (`1`..`7`, with an
accidental such as `b3` for notes outside it). `numeral` is the Roman numeral
of the triad built on that degree — `I ii iii IV V vi vii*` in major,
`i ii* III iv v VI VII` in minor — and is `null` for a chromatic note, which
has no diatonic triad.

The key is estimated by correlating a pitch-class histogram, weighted by note
duration and confidence, against the Krumhansl-Kessler profiles. `fit` is that
correlation, `margin` is how far clear of the runner-up key it is, and
`in_key` is the share of played time whose pitch belongs to the scale —
around 0.58 is what random pitches would score, so treat anything near that as
no better than a guess.

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
- Stage 6 (fretboard mapping) — built, part of `fretwise analyze`
- Stage 7 (slowed renders) — built: `fretwise render --speeds 0.5,0.75,1.0`
- Key estimation and scale degrees — built, reported by `fretwise analyze`
- Stage 8 (practice view) — built: `fretwise view`

## Using it

```
fretwise ingest "<youtube url>" --end 150   # download and trim
fretwise separate --all                     # isolate the guitar stem
fretwise analyze --separate                 # notes, key, degrees, fretboard
fretwise render                             # 50% / 75% / 100% audio
fretwise view --title "Some Song"           # work/practice.html
```

Serving the page needs a web server that honours HTTP `Range`; without it a
browser treats the audio as unseekable and the timeline cannot scrub.
GitHub Pages does. `python -m http.server` does **not**.

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
