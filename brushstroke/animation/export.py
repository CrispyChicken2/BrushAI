"""Video export through ffmpeg: raw RGB frames piped to stdin.

MP4 (H.264, yuv420p, CRF) and WebM (VP9) are encoded in streaming mode, simultaneously from the
same frame iterator. GIF uses the classic two-pass palette workflow (``palettegen`` then
``paletteuse``): the frames are spooled once to a temporary raw file so the encoder can read them
twice, with optional down-scaling/fps reduction and dithering.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import IO

import numpy as np

from brushstroke.animation.config import AnimationConfig

FORMATS = ("mp4", "webm", "gif")
GIF_DITHERS = ("none", "bayer", "sierra2_4a", "floyd_steinberg")


def find_ffmpeg(name: str = "ffmpeg") -> str:
    """Path of the ffmpeg executable, or a ``RuntimeError`` explaining how to install it."""
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"ffmpeg executable {name!r} not found on PATH. Install ffmpeg "
            "(Windows: `winget install Gyan.FFmpeg`; Linux: `apt install ffmpeg`; macOS: "
            "`brew install ffmpeg`) or set animation.ffmpeg to its full path."
        )
    return path


def _input_args(width: int, height: int, fps: int, source: str) -> list[str]:
    return [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        source,
    ]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}"
        )


def _even_scale() -> str:
    return "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # yuv420p needs even dimensions


def stream_command(
    fmt: str, ffmpeg: str, size: tuple[int, int], cfg: AnimationConfig, out: Path
) -> list[str]:
    """ffmpeg command reading raw rgb24 frames from stdin for the streaming formats."""
    width, height = size
    base = [ffmpeg, "-y", "-loglevel", "error", *_input_args(width, height, cfg.fps, "-"), "-an"]
    if fmt == "mp4":
        return [
            *base,
            "-vf",
            _even_scale(),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(cfg.mp4_crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ]
    if fmt == "webm":
        return [
            *base,
            "-vf",
            _even_scale(),
            "-c:v",
            "libvpx-vp9",
            "-crf",
            str(cfg.webm_crf),
            "-b:v",
            "0",
            "-row-mt",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
    raise ValueError(f"not a streaming format: {fmt!r}")


def encode_gif(
    raw: Path, size: tuple[int, int], cfg: AnimationConfig, out: Path, ffmpeg: str
) -> None:
    """Two-pass GIF (palettegen + paletteuse) from a raw rgb24 file."""
    width, height = size
    gif_w = min(cfg.gif_width, width) if cfg.gif_width > 0 else width
    chain = f"fps={cfg.gif_fps},scale={gif_w}:-1:flags=lanczos"
    dither = cfg.gif_dither
    paint = f"paletteuse=dither={dither}" + (":bayer_scale=5" if dither == "bayer" else "")
    with tempfile.TemporaryDirectory() as tmp:
        palette = Path(tmp) / "palette.png"
        src = _input_args(width, height, cfg.fps, str(raw))
        _run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                *src,
                "-vf",
                f"{chain},palettegen=stats_mode=diff",
                "-frames:v",
                "1",
                "-update",
                "1",
                str(palette),
            ]
        )
        _run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                *src,
                "-i",
                str(palette),
                "-lavfi",
                f"{chain}[x];[x][1:v]{paint}",
                "-loop",
                "0",
                str(out),
            ]
        )


def export_frames(
    frames: Iterable[np.ndarray], out_base: str | Path, size: tuple[int, int], cfg: AnimationConfig
) -> dict[str, Path]:
    """Encode ``(H, W, 3)`` uint8 frames into every format of ``cfg.formats``.

    Args:
        frames: iterable of frames, all of shape ``(size[1], size[0], 3)``.
        out_base: output path without extension (``<out_base>.mp4`` etc.).
        size: ``(width, height)`` of the frames.
        cfg: fps, formats, CRFs, GIF settings.

    Returns:
        Mapping format -> written file.

    Raises:
        RuntimeError: ffmpeg is missing or fails.
        ValueError: unknown format.
    """
    formats = list(dict.fromkeys(cfg.formats))
    unknown = [f for f in formats if f not in FORMATS]
    if unknown or cfg.gif_dither not in GIF_DITHERS:
        raise ValueError(f"formats must be within {FORMATS} and gif_dither within {GIF_DITHERS}")
    ffmpeg = find_ffmpeg(cfg.ffmpeg)
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = {fmt: base.with_suffix(f".{fmt}") for fmt in formats}
    width, height = size
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "frames.rgb"
        raw = raw_path.open("wb") if "gif" in formats else None
        procs: dict[str, subprocess.Popen[bytes]] = {}
        errs: dict[str, IO[bytes]] = {}
        try:
            for fmt in formats:
                if fmt == "gif":
                    continue
                errs[fmt] = (Path(tmp) / f"{fmt}.err").open("wb")
                procs[fmt] = subprocess.Popen(
                    stream_command(fmt, ffmpeg, size, cfg, outputs[fmt]),
                    stdin=subprocess.PIPE,
                    stderr=errs[fmt],
                )
            for frame in frames:
                if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                    raise ValueError(
                        f"frame must be uint8 {(height, width, 3)}, got {frame.dtype} {frame.shape}"
                    )
                data = np.ascontiguousarray(frame).tobytes()
                for p in procs.values():
                    assert p.stdin is not None
                    p.stdin.write(data)
                if raw is not None:
                    raw.write(data)
        finally:
            for p in procs.values():
                if p.stdin is not None:
                    p.stdin.close()
            codes = {fmt: p.wait() for fmt, p in procs.items()}
            for f in errs.values():
                f.close()
            if raw is not None:
                raw.close()
        for fmt, code in codes.items():
            if code != 0:
                msg = (Path(tmp) / f"{fmt}.err").read_text(errors="replace")[-2000:]
                raise RuntimeError(f"ffmpeg ({fmt}) exited with {code}:\n{msg}")
        if "gif" in formats:
            encode_gif(raw_path, size, cfg, outputs["gif"], ffmpeg)
    return outputs
