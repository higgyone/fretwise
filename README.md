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

- `--fmin` / `--fmax` — pitch search range (default guitar range: E2–E6)
- `--separate` — on/off for Demucs source separation
- `--onset-threshold` / `--frame-threshold` — how much the transcriber keeps
- `--held-gap` — join same-pitch notes closer than this into one held note
- `--max-stray` — drop notes this far above the line around them
- `--engine` — `basic-pitch` (polyphonic, default) or `pyin` (one pitch at a time)

## Stack

- **Python**: `yt-dlp`, `basic-pitch`, `demucs`, `librosa`, `numpy`, ffmpeg
- **Web**: single static HTML file, embedded JS, SVG fretboard — no build
  tooling required. Deployable as-is via GitHub Pages, so practice sessions
  work from a phone browser once pushed.

## Status

- Stage 1 (ingest) — built: `fretwise ingest <url or file> --start 1:23 --end 1:41`
- Stage 2 (separate) — built: `fretwise separate --all`
- Stages 3-5 (onsets, pitch, note names) — built: `fretwise analyze --separate`
- Stage 6 (fretboard mapping) — built, part of `fretwise analyze`
- Stage 7 (slowed renders) — built: `fretwise render --speeds 0.5,0.75,1.0`
- Key estimation and scale degrees — built, reported by `fretwise analyze`
- Stage 8 (practice view) — built: `fretwise view`
- Confidence highlighting — built, shown on the fretboard dots
- ASCII tab export — built: `fretwise tab --start 0 --end 40`
- Re-listening to doubtful notes — built: `fretwise review`

## Setup

Once, in the project directory:

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

On Linux or macOS that is `.venv/bin/python` throughout. Every command below
is written as `fretwise`, meaning `.venv/Scripts/python.exe -m fretwise`.

No system ffmpeg is needed: `imageio-ffmpeg` supplies one if none is on
`PATH`. The first `separate` downloads the Demucs model, a few hundred MB.

## From a YouTube URL to a practice page

Five commands. Each writes into `work/`, so later steps pick up what earlier
ones left.

```
fretwise ingest "https://www.youtube.com/watch?v=..." --start 0 --end 150
fretwise separate --all
fretwise analyze --separate
fretwise render
fretwise view --title "Some Song"
```

1. **ingest** downloads the audio, trims it to `--start`/`--end` (`12`,
   `1:23` and `0:01:23.5` all work; omit `--end` for the whole thing) and
   writes `work/clip.wav`. The download is cached, so changing the trim
   points does not fetch it again.
2. **separate** splits the mix into stems, writing `work/clip-guitar.wav`
   and the rest. This is the slow step, around 90s for a 150s clip on a CPU.
   `--all` keeps every stem so you can check which one your part landed in.
3. **analyze** transcribes the guitar stem into `work/notes.json`: notes with
   times, durations, confidences, the estimated key, scale degrees, and a
   string and fret for each note.
4. **render** writes `work/clip-50.wav`, `-75` and `-100`, pitch-preserving
   slowed copies for the page's speed selector.
5. **view** writes `work/practice.html`, which reads the audio beside it.

`fretwise tab` writes the same part as ASCII tablature to `work/tab.txt`,
for printing or pasting somewhere. Spacing there is proportional to elapsed
time rather than laid out on a beat grid, since nothing here detects tempo.

## From an audio file you already have

`ingest` takes a path as readily as a URL, and anything ffmpeg can read
works — `.wav`, `.mp3`, `.m4a`, `.flac`, even a video file:

```
fretwise ingest "C:/music/riff.mp3" --start 0:30 --end 1:10
```

Everything after that is identical. If your file is already a mono WAV of
just the part you want, you can skip `ingest` by copying it to
`work/clip.wav` yourself.

## Opening the page

The page needs its `clip-*.wav` files in the same folder. Opening
`work/practice.html` directly from disk works.

**To serve it, the server must honour HTTP `Range`.** Without that a browser
treats the audio as unseekable, and the timeline silently refuses to scrub —
which looks like a bug in the page. GitHub Pages honours it. Python's
`http.server` does **not**.

Copied somewhere without its audio, the page says so and still works for
reading the part: the fretboard and timeline run from a clock of their own.

## Using the page

- **Play/pause** — the button, or space
- **Step note to note** — the `◀ note` / `note ▶` buttons, or left and right
  arrows. Stepping pauses playback, so you can work a phrase out one note at
  a time. A strummed chord counts as one step, not one per string.
- **Jump 5 seconds** — shift with the arrow keys
- **Speed** — 50%, 75% or 100%, keeping your place when you switch
- **Loop** — drag across the timeline; click it to seek
- **Confidence** — dots are shaded by how strongly the note was detected,
  relative to this clip. Faint and hollow means less certain than what is
  around it. `Hide below` removes the weakest, and where to draw that line is
  your call: nothing in the numbers marks a natural cut-off.

## When the result is poor

- **Wrong instrument.** `htdemucs_6s` has a real guitar stem, but a part can
  still land in `bass` or `other`. Run `fretwise separate --all` and listen to
  the stems, then point analysis at the right one:
  `fretwise analyze work/clip-bass.wav`.
- **Notes missing once the band comes in.** Separation can leave the guitar
  stem nearly empty in dense passages — on one clip it held under 2% of the
  energy while drums, bass and vocals took the rest — and nothing can be
  transcribed from silence. `fretwise analyze --separate --stems guitar,bass`
  adds the stems together, which recovers the playing where it went. Some
  genuine bass notes come with it, so it is a choice rather than a default.
- **Too few or too many notes.** `--onset-threshold` (lower finds more note
  starts) and `--frame-threshold` (lower keeps quieter notes).
- **Notes stuttering.** `--held-gap` joins same-pitch notes closer than that
  many seconds into one held note.
- **A note wildly out of range.** `--max-stray` drops notes that many
  semitones above the line around them; `0` disables it.
- **Check by ear.** `fretwise sonify --start 0 --end 40` plays the detected
  notes over the original, which finds gross errors no number reveals.
  `fretwise probe` reports every onset and why it was kept or rejected.
- **Check the doubtful ones.** `fretwise review` builds a short file that
  plays, for each of the least certain notes, the recording around it
  followed by the note it was read as, with an index of where each sits in
  the original. Whether a quiet note is real is a question for your ear.

## Roadmap / stretch goals

All of the original spec and its stretch goals are built. What remains is
listed under "Known limitations" in the pull requests: the fretboard mapping
is greedy and only aims to be playable rather than optimal, and a real octave
doubling still cannot be told from a spurious one by any measurement tried.
