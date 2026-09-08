"""Command line entry point for the fretwise pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter

import numpy as np
from pathlib import Path

from . import sonify
from .analyze import (
    ANALYSIS_SAMPLE_RATE, MIN_CONFIDENCE, RANGE_PADDING_SEMITONES, Candidate,
    analyze_file, candidates_file, load_audio, load_notes,
)
from .ingest import DEFAULT_SAMPLE_RATE, IngestError, ingest
from .fretboard import DEFAULT_MAX_FRET, map_notes
from . import review, tab
from .stretch import DEFAULT_SPEEDS, StretchError, render_speeds
from .view import DEFAULT_VIEW_SPEED, ViewError, write_page
from .cleanup import MAX_HELD_GAP, MAX_SEMITONES_ABOVE, drop_stray_notes, merge_held_notes
from .key import annotate, estimate_key, in_key_fraction
from .notes import NoteError, midi_to_hz, name_to_midi
from .transcribe import (
    DEFAULT_FRAME_THRESHOLD, DEFAULT_MAX_MIDI, DEFAULT_MIN_MIDI, DEFAULT_ONSET_THRESHOLD,
    TranscriptionError, transcribe_file,
)
from .separate import (
    DEFAULT_MODEL, DEFAULT_STEM, STEMS, SeparationError, separate, separate_all,
)
from .timestamps import TimestampError, format_timestamp, parse_timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fretwise", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = subcommands.add_parser(
        "ingest", help="download a clip and trim it to a mono WAV"
    )
    ingest_cmd.add_argument(
        "source",
        help="a URL to download, or the path of an audio or video file on disk",
    )
    ingest_cmd.add_argument(
        "--start", default=None, help="clip start, e.g. 12, 1:23 or 0:01:23.5"
    )
    ingest_cmd.add_argument("--end", default=None, help="clip end, same formats as --start")
    ingest_cmd.add_argument(
        "-o", "--work-dir", default="work", type=Path, help="output directory (default: work)"
    )
    ingest_cmd.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="output sample rate"
    )
    ingest_cmd.add_argument(
        "--force", action="store_true", help="re-download even if the source is cached"
    )
    separate_cmd = subcommands.add_parser(
        "separate", help="split a clip into stems and keep one (needs demucs)"
    )
    separate_cmd.add_argument(
        "clip", nargs="?", default=None, type=Path,
        help="WAV to separate (default: <work-dir>/clip.wav)",
    )
    separate_cmd.add_argument(
        "-o", "--work-dir", default="work", type=Path, help="working directory (default: work)"
    )
    separate_cmd.add_argument(
        "--stem", default=DEFAULT_STEM, choices=STEMS,
        help=f"stem to keep; lead guitar lands in 'other' (default: {DEFAULT_STEM})",
    )
    separate_cmd.add_argument("--model", default=DEFAULT_MODEL, help="demucs model name")
    separate_cmd.add_argument(
        "--all", action="store_true",
        help="keep every stem, not just --stem (free: demucs computes them all anyway)",
    )

    analyze_cmd = subcommands.add_parser(
        "analyze", help="find the notes in a clip and write notes.json"
    )
    analyze_cmd.add_argument(
        "clip", nargs="?", default=None, type=Path,
        help="WAV to analyze (default: <work-dir>/clip.wav)",
    )
    analyze_cmd.add_argument(
        "-o", "--work-dir", default="work", type=Path, help="working directory (default: work)"
    )
    analyze_cmd.add_argument(
        "--max-fret", type=int, default=DEFAULT_MAX_FRET,
        help=f"highest fret to map notes onto (default: {DEFAULT_MAX_FRET})",
    )
    analyze_cmd.add_argument(
        "--held-gap", type=float, default=MAX_HELD_GAP,
        help="join same-pitch notes closer than this, as one held note "
             f"(default: {MAX_HELD_GAP}s; 0 disables)",
    )
    analyze_cmd.add_argument(
        "--max-stray", type=int, default=MAX_SEMITONES_ABOVE,
        help="drop notes this many semitones above the line around them "
             f"(default: {MAX_SEMITONES_ABOVE}; 0 disables)",
    )
    analyze_cmd.add_argument(
        "--engine", default="basic-pitch", choices=("basic-pitch", "pyin"),
        help="basic-pitch transcribes chords; pyin tracks one pitch at a time",
    )
    analyze_cmd.add_argument(
        "--onset-threshold", type=float, default=DEFAULT_ONSET_THRESHOLD,
        help="basic-pitch: lower finds more note starts (default: 0.5)",
    )
    analyze_cmd.add_argument(
        "--frame-threshold", type=float, default=DEFAULT_FRAME_THRESHOLD,
        help="basic-pitch: lower keeps quieter notes (default: 0.3)",
    )
    analyze_cmd.add_argument(
        "--sensitivity", type=float, default=0.5,
        help="onset sensitivity 0..1; higher finds more notes (default: 0.5)",
    )
    analyze_cmd.add_argument(
        "--fmin", default="E2", help="lowest pitch to search for (note name or Hz)"
    )
    analyze_cmd.add_argument(
        "--fmax", default="E6", help="highest pitch to search for (note name or Hz)"
    )
    analyze_cmd.add_argument(
        "--separate", action="store_true",
        help="isolate a stem with demucs first (recommended for full mixes)",
    )
    analyze_cmd.add_argument(
        "--stem", default=DEFAULT_STEM, choices=STEMS, help="stem to analyze with --separate"
    )
    analyze_cmd.add_argument(
        "--min-confidence", type=float, default=None,
        help="drop notes below this confidence (0..1)",
    )
    probe_cmd = subcommands.add_parser(
        "probe", help="report every onset and why it was kept or rejected"
    )
    probe_cmd.add_argument(
        "clip", nargs="?", default=None, type=Path,
        help="WAV to probe (default: the separated stem, else <work-dir>/clip.wav)",
    )
    probe_cmd.add_argument(
        "-o", "--work-dir", default="work", type=Path, help="working directory (default: work)"
    )
    probe_cmd.add_argument(
        "--sensitivity", type=float, default=0.5, help="onset sensitivity 0..1 (default: 0.5)"
    )
    probe_cmd.add_argument(
        "--min-confidence", type=float, default=None, help="confidence floor to test (0..1)"
    )
    probe_cmd.add_argument("--fmin", default="E2", help="lowest pitch to search for")
    probe_cmd.add_argument("--fmax", default="E6", help="highest pitch to search for")

    review_cmd = subcommands.add_parser(
        "review", help="hear the least certain notes against the recording"
    )
    review_cmd.add_argument(
        "-o", "--work-dir", default="work", type=Path, help="working directory (default: work)"
    )
    review_cmd.add_argument(
        "--count", type=int, default=review.DEFAULT_COUNT,
        help=f"how many of the weakest notes to review (default: {review.DEFAULT_COUNT})",
    )
    review_cmd.add_argument(
        "--below", type=float, default=None,
        help="review every note under this confidence instead of a fixed count",
    )
    review_cmd.add_argument(
        "--source", default="stem", choices=("stem", "mix"),
        help="listen to the separated guitar (default) or the whole mix",
    )
    review_cmd.add_argument(
        "--pad", type=float, default=review.PAD,
        help=f"seconds of recording either side of each note (default: {review.PAD})",
    )

    tab_cmd = subcommands.add_parser(
        "tab", help="write the part as ASCII guitar tablature"
    )
    tab_cmd.add_argument(
        "-o", "--work-dir", default="work", type=Path, help="working directory (default: work)"
    )
    tab_cmd.add_argument("--start", default=None, help="only from this time")
    tab_cmd.add_argument("--end", default=None, help="only up to this time")
    tab_cmd.add_argument("--title", default=None, help="heading above the tab")
    tab_cmd.add_argument(
        "--width", type=int, default=tab.SYSTEM_WIDTH,
        help=f"characters per line (default: {tab.SYSTEM_WIDTH})",
    )
    tab_cmd.add_argument(
        "--seconds-per-column", type=float, default=tab.SECONDS_PER_COLUMN,
        help=f"how much time one column stands for (default: {tab.SECONDS_PER_COLUMN})",
    )
    tab_cmd.add_argument(
        "--max-rest", type=int, default=tab.MAX_REST_COLUMNS,
        help=f"most columns a silence may take (default: {tab.MAX_REST_COLUMNS})",
    )

    view_cmd = subcommands.add_parser(
        "view", help="build the practice page: fretboard, playhead, speeds, looping"
    )
    view_cmd.add_argument(
        "-o", "--work-dir", default="work", type=Path, help="working directory (default: work)"
    )
    view_cmd.add_argument("--title", default=None, help="heading for the page")
    view_cmd.add_argument(
        "--default-speed", type=float, default=DEFAULT_VIEW_SPEED,
        help=f"speed the page opens at (default: {DEFAULT_VIEW_SPEED})",
    )
    view_cmd.add_argument(
        "--max-fret", type=int, default=DEFAULT_MAX_FRET, help="frets to draw"
    )

    render_cmd = subcommands.add_parser(
        "render", help="pre-render the clip at slower, pitch-preserving speeds"
    )
    render_cmd.add_argument(
        "clip", nargs="?", default=None, type=Path,
        help="WAV to render (default: <work-dir>/clip.wav)",
    )
    render_cmd.add_argument(
        "-o", "--work-dir", default="work", type=Path, help="working directory (default: work)"
    )
    render_cmd.add_argument(
        "--speeds", default=",".join(str(s) for s in DEFAULT_SPEEDS),
        help=f"comma separated playback speeds (default: {','.join(str(s) for s in DEFAULT_SPEEDS)})",
    )
    render_cmd.add_argument(
        "--backend", default="ffmpeg", choices=("ffmpeg", "librosa"),
        help="ffmpeg's atempo (default) or librosa's phase vocoder",
    )

    sonify_cmd = subcommands.add_parser(
        "sonify", help="play the detected notes back, to check them by ear"
    )
    sonify_cmd.add_argument(
        "-o", "--work-dir", default="work", type=Path, help="working directory (default: work)"
    )
    sonify_cmd.add_argument("--start", default=None, help="only render from this time")
    sonify_cmd.add_argument("--end", default=None, help="only render up to this time")
    sonify_cmd.add_argument(
        "--sample-rate", type=int, default=22050, help="output sample rate (default: 22050)"
    )
    sonify_cmd.add_argument(
        "--clicks", action="store_true",
        help="mark every detected onset with a click, from probe output",
    )
    sonify_cmd.add_argument(
        "--all", action="store_true",
        help="also play onsets rejected by the confidence floor, from probe output",
    )
    return parser


def parse_pitch(value: str, *, padding: int = 0) -> float:
    """Accept either a note name (``E2``) or a bare frequency in Hz.

    ``padding`` shifts a named pitch by that many semitones, so a requested
    bound stays inside the range pyin actually searches.
    """
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return midi_to_hz(name_to_midi(value) + padding)
    except NoteError:
        raise NoteError(f"not a note name or frequency: {value!r}") from None


def run_ingest(args: argparse.Namespace) -> int:
    clip = ingest(
        args.source,
        work_dir=args.work_dir,
        start=args.start,
        end=args.end,
        sample_rate=args.sample_rate,
        force=args.force,
    )
    span = format_timestamp(clip.start)
    if clip.end is not None:
        span += f" - {format_timestamp(clip.end)} ({clip.duration:.2f}s)"
    else:
        span += " - end of source"
    print(f"{clip.title}\n  {span}\n  {clip.path} @ {clip.sample_rate} Hz mono")
    return 0


def resolve_clip(args: argparse.Namespace) -> Path:
    clip = args.clip or args.work_dir / "clip.wav"
    if not clip.exists():
        raise IngestError(f"no clip at {clip} — run `fretwise ingest` first")
    return clip


def run_separate(args: argparse.Namespace) -> int:
    clip = resolve_clip(args)
    print(f"separating {clip} with {args.model} (this takes a while on CPU)...")
    if args.all:
        for name, path in sorted(separate_all(clip, model=args.model, out_dir=args.work_dir).items()):
            print(f"  {name:7} -> {path}")
        return 0
    stem_path = separate(clip, stem=args.stem, model=args.model, out_dir=args.work_dir)
    print(f"{args.stem} stem -> {stem_path}")
    return 0


def run_analyze(args: argparse.Namespace) -> int:
    clip = resolve_clip(args)
    if args.separate:
        stem_path = args.work_dir / f"{clip.stem}-{args.stem}.wav"
        if stem_path.exists():
            print(f"using cached {args.stem} stem: {stem_path}")
        else:
            print(f"separating {clip} (this takes a while on CPU)...")
            stem_path = separate(clip, stem=args.stem, out_dir=args.work_dir)
        clip = stem_path

    options = {
        "sensitivity": args.sensitivity,
        "fmin": parse_pitch(args.fmin, padding=-RANGE_PADDING_SEMITONES),
        "fmax": parse_pitch(args.fmax, padding=RANGE_PADDING_SEMITONES),
    }
    if args.min_confidence is not None:
        options["min_confidence"] = args.min_confidence

    if args.engine == "basic-pitch":
        notes = transcribe_file(
            clip,
            min_midi=name_to_midi(args.fmin) if not args.fmin[0].isdigit() else DEFAULT_MIN_MIDI,
            max_midi=name_to_midi(args.fmax) if not args.fmax[0].isdigit() else DEFAULT_MAX_MIDI,
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold,
        )
    else:
        notes = analyze_file(clip, **options)
    if args.held_gap:
        before = len(notes)
        # The recording decides whether a boundary is a strum or a split, so
        # the merge needs to hear what the transcriber heard.
        heard, heard_sr = load_audio(clip, sample_rate=ANALYSIS_SAMPLE_RATE)
        notes = merge_held_notes(
            notes, max_gap=args.held_gap, audio=heard, sr=heard_sr
        )
        if before != len(notes):
            print(f"joined {before - len(notes)} fragments into held notes")

    if args.max_stray:
        before = len(notes)
        notes = drop_stray_notes(notes, max_semitones_above=args.max_stray)
        if before != len(notes):
            print(f"dropped {before - len(notes)} notes far above the surrounding line")

    key = estimate_key(notes)
    annotate(notes, key)
    mapping: dict = {}
    notes = map_notes(notes, max_fret=args.max_fret, stats=mapping)
    if mapping.get("shortened"):
        print(
            f"{mapping['shortened']} notes shortened where a later note "
            "needed their string"
        )
    if mapping.get("unplaced"):
        print(f"{mapping['unplaced']} notes could not be given a string at all")

    destination = args.work_dir / "notes.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": None if key is None else {
            "name": key.name,
            "tonic": key.tonic,
            "mode": key.mode,
            "fit": round(key.fit, 4),
            "margin": round(key.margin, 4),
            "in_key": round(in_key_fraction(notes, key), 4),
        },
        "notes": [note.to_dict() for note in notes],
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{len(notes)} notes -> {destination}")
    if key is not None:
        print(
            f"key: {key.name}  (fit {key.fit:.2f}, "
            f"{100 * in_key_fraction(notes, key):.0f}% of played time in key, "
            f"{key.margin:.2f} clear of next best)"
        )
    for note in notes[:10]:
        degree = f"{note.degree:>3}" if note.degree else "  -"
        where = (
            f"str {note.chosen['string']} fret {note.chosen['fret']:<2}"
            if note.chosen else "unplayable"
        )
        print(
            f"  {format_timestamp(note.time)}  {note.note:<4}"
            f"  {note.duration:5.2f}s  conf {note.confidence:.2f}"
            f"  degree {degree:<3} {note.numeral or '-':<5} {where}"
        )
    if len(notes) > 10:
        print(f"  ... and {len(notes) - 10} more")
    return 0


def stem_or_clip(args: argparse.Namespace) -> Path:
    """Prefer a separated stem when one has been made; fall back to the raw clip."""
    if getattr(args, "clip", None):
        return resolve_clip(args)
    stem = args.work_dir / f"clip-{DEFAULT_STEM}.wav"
    return stem if stem.exists() else resolve_clip(args)


def run_probe(args: argparse.Namespace) -> int:
    import csv

    clip = stem_or_clip(args)
    options = {
        "sensitivity": args.sensitivity,
        "fmin": parse_pitch(args.fmin, padding=-RANGE_PADDING_SEMITONES),
        "fmax": parse_pitch(args.fmax, padding=RANGE_PADDING_SEMITONES),
    }
    if args.min_confidence is not None:
        options["min_confidence"] = args.min_confidence

    print(f"probing {clip}...")
    found = candidates_file(clip, **options)
    if not found:
        print("no onsets found")
        return 0

    rows = [c.to_dict() for c in found]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "onsets.json").write_text(json.dumps(rows, indent=2) + chr(10))
    with open(args.work_dir / "onsets.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    span = max(c.time + c.duration for c in found)
    verdicts = Counter(c.verdict for c in found)
    print(f"{len(found)} onsets over {span:.1f}s ({len(found) / span * 60:.0f} per minute)")
    for verdict, count in verdicts.most_common():
        print(f"  {verdict:<16} {count:4d}  ({100 * count / len(found):4.1f}%)")

    scored = [c.confidence for c in found if c.hz is not None]
    if scored:
        percentiles = np.percentile(scored, [50, 75, 90, 99])
        print(
            "  confidence of pitched onsets: "
            + "  ".join(f"p{p}={v:.3f}" for p, v in zip((50, 75, 90, 99), percentiles))
        )
        floor = args.min_confidence if args.min_confidence is not None else MIN_CONFIDENCE
        for trial in (floor, floor / 2, floor / 5):
            passing = sum(1 for c in scored if c >= trial)
            print(f"  at a floor of {trial:.3f}: {passing:4d} notes ({passing / span * 60:.0f}/min)")

    print(f"  wrote {args.work_dir / 'onsets.csv'} and onsets.json")
    return 0


def load_probe(work_dir: Path) -> list[Candidate]:
    """Read the candidates written by `fretwise probe`."""
    path = work_dir / "onsets.json"
    if not path.exists():
        raise IngestError(f"no probe output at {path} - run `fretwise probe` first")
    return [Candidate(**entry) for entry in json.loads(path.read_text(encoding="utf-8"))]


def run_review(args: argparse.Namespace) -> int:
    notes_path = args.work_dir / "notes.json"
    if not notes_path.exists():
        raise ViewError(f"no notes at {notes_path} - run `fretwise analyze` first")

    notes, _key = load_notes(notes_path)
    doubtful = review.weakest(notes, count=args.count, below=args.below)
    if not doubtful:
        print("no notes matched")
        return 0

    sr = 22050
    # The stem is what the transcriber actually heard, and a reference tone is
    # far easier to judge against one instrument than against the whole band.
    stem = args.work_dir / f"clip-{DEFAULT_STEM}.wav"
    source = stem if args.source == "stem" and stem.exists() else args.work_dir / "clip.wav"
    clip, _ = load_audio(source, sample_rate=sr)
    audio, items = review.build(clip, sr, doubtful, context=notes, pad=args.pad)
    destination = sonify.write(audio, sr, args.work_dir / "review.wav")

    print(review.index(items))
    print(f"{len(items)} notes from {source.name}, {len(audio) / sr:.0f}s -> {destination}")
    print("the recording at each doubtful moment, with the transcription")
    print("played over it, exactly as `fretwise sonify` does")
    return 0


def run_tab(args: argparse.Namespace) -> int:
    notes_path = args.work_dir / "notes.json"
    if not notes_path.exists():
        raise ViewError(f"no notes at {notes_path} - run `fretwise analyze` first")

    notes, key = load_notes(notes_path)
    start = parse_timestamp(args.start) or 0.0
    end = parse_timestamp(args.end)
    chosen = [n for n in notes if n.time >= start and (end is None or n.time < end)]

    text = tab.header(chosen, key, title=args.title or "") + tab.render(
        chosen,
        width=args.width,
        seconds_per_column=args.seconds_per_column,
        max_rest=args.max_rest,
    )

    destination = args.work_dir / "tab.txt"
    destination.write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {destination}")
    return 0


def run_view(args: argparse.Namespace) -> int:
    notes_path = args.work_dir / "notes.json"
    if not notes_path.exists():
        raise ViewError(f"no notes at {notes_path} - run `fretwise analyze` first")

    notes, key = load_notes(notes_path)
    page = write_page(
        notes, args.work_dir, key=key,
        title=args.title or notes_path.parent.resolve().name,
        default_speed=args.default_speed, max_fret=args.max_fret,
    )

    speeds = ", ".join(f"{speed:g}x" for speed in sorted(page.speeds))
    print(f"{page.path}  ({page.notes} notes, speeds: {speeds or 'none'})")
    if page.missing_audio:
        print("  missing audio, so those speeds are not offered: "
              + ", ".join(page.missing_audio))
        print("  run `fretwise render` to make them")
    return 0


def run_render(args: argparse.Namespace) -> int:
    clip = resolve_clip(args)
    try:
        speeds = tuple(float(part) for part in args.speeds.split(",") if part.strip())
    except ValueError:
        raise StretchError(f"could not read speeds: {args.speeds!r}") from None

    print(f"rendering {clip} at {', '.join(f'{s:g}x' for s in speeds)}...")
    for speed, path in sorted(render_speeds(
        clip, speeds=speeds, out_dir=args.work_dir, backend=args.backend
    ).items()):
        print(f"  {speed:>5g}x -> {path}")
    return 0


def run_sonify(args: argparse.Namespace) -> int:
    start = parse_timestamp(args.start) or 0.0
    end = parse_timestamp(args.end)

    def within(time: float) -> bool:
        return time >= start and (end is None or time < end)

    key = None
    if args.all:
        found = load_probe(args.work_dir)
        played = [c.to_note() for c in found if within(c.time) and c.hz is not None]
        onsets = [c.time - start for c in found if within(c.time)]
        label = f"{len(played)} pitched onsets (confidence floor ignored)"
    else:
        notes_path = args.work_dir / "notes.json"
        if not notes_path.exists():
            raise IngestError(f"no notes at {notes_path} - run `fretwise analyze` first")
        notes, key = load_notes(notes_path)
        played = [n for n in notes if within(n.time)]
        onsets = [c.time - start for c in load_probe(args.work_dir) if within(c.time)] if args.clicks else []
        label = f"{len(played)} notes"

    for note in played:  # shift so the excerpt starts at zero
        note.time -= start
    if not played:
        print("no notes in that range")
        return 0

    sr = args.sample_rate
    clip, _ = load_audio(args.work_dir / "clip.wav", sample_rate=sr)
    clip = clip[int(start * sr) : None if end is None else int(end * sr)]
    duration = len(clip) / sr

    rendered = sonify.render(played, sr, duration=duration)
    if args.clicks:
        rendered = sonify.normalize(rendered + sonify.render_clicks(onsets, sr, duration) * 0.6)

    suffix = "-all" if args.all else ""
    notes_only = sonify.write(rendered, sr, args.work_dir / f"notes{suffix}.wav")
    mixed = sonify.write(
        sonify.mix(clip, rendered), sr, args.work_dir / f"notes{suffix}-mix.wav"
    )

    span = f"{format_timestamp(start)} to " + (format_timestamp(end) if end else "the end")
    print(f"{label}, {span}" + (f", key {key['name']}" if key else ""))
    if args.clicks:
        print(f"  {len(onsets)} onsets marked with clicks")
    print(f"  detected notes alone: {notes_only}")
    print(f"  over the original:    {mixed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "ingest":
            return run_ingest(args)
        if args.command == "analyze":
            return run_analyze(args)
        if args.command == "separate":
            return run_separate(args)
        if args.command == "sonify":
            return run_sonify(args)
        if args.command == "render":
            return run_render(args)
        if args.command == "view":
            return run_view(args)
        if args.command == "tab":
            return run_tab(args)
        if args.command == "review":
            return run_review(args)
        if args.command == "probe":
            return run_probe(args)
    except (
        IngestError, TimestampError, NoteError, SeparationError, TranscriptionError,
        StretchError, ViewError, ValueError
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
