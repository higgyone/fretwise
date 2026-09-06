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
    MIN_CONFIDENCE, RANGE_PADDING_SEMITONES, Candidate, analyze_file, candidates_file,
    load_audio, load_notes,
)
from .ingest import DEFAULT_SAMPLE_RATE, IngestError, ingest
from .key import annotate, estimate_key, in_key_fraction
from .notes import NoteError, midi_to_hz, name_to_midi
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
    ingest_cmd.add_argument("url", help="YouTube (or any yt-dlp supported) URL")
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
        args.url,
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

    notes = analyze_file(clip, **options)
    key = estimate_key(notes)
    annotate(notes, key)

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
        print(
            f"  {format_timestamp(note.time)}  {note.note:<4}"
            f"  {note.duration:5.2f}s  conf {note.confidence:.2f}"
            f"  degree {degree}  {note.numeral or '-'}"
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
        if args.command == "probe":
            return run_probe(args)
    except (IngestError, TimestampError, NoteError, SeparationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
