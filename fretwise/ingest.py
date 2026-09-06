"""Stage 1 — ingest.

Downloads audio for a YouTube URL with ``yt-dlp``, then uses ``ffmpeg`` to trim
to a start/end range and normalize to a mono WAV that the later analysis stages
can load directly.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .timestamps import parse_timestamp

DEFAULT_SAMPLE_RATE = 44100


class IngestError(RuntimeError):
    """Raised when a download or ffmpeg step fails."""


@dataclass
class Clip:
    """A trimmed, normalized audio clip ready for analysis."""

    path: Path
    source_url: str
    title: str
    start: float
    end: float | None
    sample_rate: int

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start


def find_ffmpeg() -> str:
    """Locate an ffmpeg binary: system PATH first, then the pip-installed one."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
    except ImportError:
        raise IngestError(
            "ffmpeg not found on PATH. Install ffmpeg, or `pip install imageio-ffmpeg`."
        ) from None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _url_slug(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def download_audio(url: str, work_dir: Path, *, force: bool = False) -> tuple[Path, str]:
    """Download the best available audio stream. Returns ``(path, title)``.

    The download is cached under ``work_dir`` keyed by URL, so re-running an
    ingest with different trim points does not re-fetch the source.
    """
    import yt_dlp

    work_dir.mkdir(parents=True, exist_ok=True)
    slug = _url_slug(url)
    stem = work_dir / f"source-{slug}"

    existing = sorted(work_dir.glob(f"source-{slug}.*"))
    options = {
        "format": "bestaudio/best",
        "outtmpl": f"{stem}.%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": str(Path(find_ffmpeg()).parent),
    }

    if existing and not force:
        # Cached: still ask yt-dlp for metadata so the title is available.
        with yt_dlp.YoutubeDL({**options, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return existing[0], info.get("title", url)

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise IngestError(f"download failed for {url}: {exc}") from exc

    downloaded = sorted(work_dir.glob(f"source-{slug}.*"))
    if not downloaded:
        raise IngestError(f"yt-dlp reported success but no file was written for {url}")
    return downloaded[0], info.get("title", url)


def trim_to_wav(
    source: Path,
    dest: Path,
    *,
    start: float = 0.0,
    end: float | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    """Trim ``source`` to ``[start, end)`` and write a mono WAV to ``dest``."""
    if end is not None and end <= start:
        raise IngestError(f"end ({end}) must be after start ({start})")

    dest.parent.mkdir(parents=True, exist_ok=True)
    # -ss before -i seeks fast; -accurate_seek keeps the cut sample-exact.
    command = [find_ffmpeg(), "-y", "-loglevel", "error", "-accurate_seek", "-ss", f"{start:.3f}"]
    if end is not None:
        command += ["-t", f"{end - start:.3f}"]
    command += [
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(dest),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise IngestError(f"ffmpeg failed ({result.returncode}): {result.stderr.strip()}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise IngestError(f"ffmpeg produced no audio at {dest}")
    return dest


def ingest(
    source: str,
    *,
    work_dir: Path | str = "work",
    start: str | float | None = None,
    end: str | float | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    force: bool = False,
) -> Clip:
    """Run stage 1 end to end: fetch, trim, normalize.

    ``source`` is either a URL to download or the path of an audio or video
    file already on disk. A local file skips the download and is trimmed and
    converted in place, so anything ffmpeg can read works as input.
    """
    work_dir = Path(work_dir)
    start_s = parse_timestamp(start) or 0.0
    end_s = parse_timestamp(end)

    local = Path(source)
    if local.exists() and local.is_file():
        source_path, title = local, local.stem
    else:
        source_path, title = download_audio(source, work_dir, force=force)
    dest = work_dir / "clip.wav"
    trim_to_wav(source_path, dest, start=start_s, end=end_s, sample_rate=sample_rate)

    return Clip(
        path=dest,
        source_url=str(source),
        title=title,
        start=start_s,
        end=end_s,
        sample_rate=sample_rate,
    )
