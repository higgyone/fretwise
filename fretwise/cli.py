"""Command line entry point for the fretwise pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import RANGE_PADDING_SEMITONES, analyze_file
from .ingest import DEFAULT_SAMPLE_RATE, IngestError, ingest
from .notes import NoteError, midi_to_hz, name_to_midi
from .separate import (
    DEFAULT_MODEL, DEFAULT_STEM, STEMS, SeparationError, separate, separate_all,
)
from .timestamps import TimestampError, format_timestamp


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
    destination = args.work_dir / "notes.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([note.to_dict() for note in notes], indent=2) + "\n"
    )

    print(f"{len(notes)} notes -> {destination}")
    for note in notes[:10]:
        print(
            f"  {format_timestamp(note.time)}  {note.note:<4}"
            f"  {note.duration:5.2f}s  conf {note.confidence:.2f}"
        )
    if len(notes) > 10:
        print(f"  ... and {len(notes) - 10} more")
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
    except (IngestError, TimestampError, NoteError, SeparationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
