"""Training subset of Learning to Paint: Met Open Access (CC0) images at 128 x 128, disjoint from the evaluation set (T-029).

Spec: ``docs/design/evaluation.md`` section 8.2 and ``docs/design/learning_to_paint.md`` section 7. The paper trained on CelebA /
ImageNet (research-only licences); here the pool is the Met Open Access objects with ``isPublicDomain = true`` and an image.

Pipeline (``scripts/build_l2p_trainset.py``): the Met's own Hugging Face export ``source/metadata_images.csv.gz`` of
``metmuseum/openaccess`` (CC0, pinned by SHA-256 and revision: the Open Access ``MetObjects.csv`` columns plus ``primaryImageSmall``, so
no call to the rate-limited Met API is needed) gives the public-domain objects with an image, their classification and artist; the
candidates are those not excluded by :func:`exclusions` (the evaluation objects, every work of an artist who has an evaluation image);
each ``primaryImageSmall`` file is downloaded, centre-cropped to a square and reduced to 128 x 128 with the Pillow box filter
(``eval.dataset`` conventions at the training size), and written to ``uint8`` shards; images that are near-duplicates of an
evaluation image (perceptual hash) are dropped at verification. Images and shards stay outside git (``data/raw/l2p_met``).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import random
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

SIZE = 128  # side of the stored images (the actor's canvas)
MIN_SHORT_SIDE = (
    256  # a source needs at least this short side (primaryImageSmall is about 1 megapixel)
)
MIN_ASPECT, MAX_ASPECT = (
    1 / 2,
    2.0,
)  # width / height; wider than the evaluation set (3:4 to 4:3), which rejected two thirds of the first candidates: the centre square keeps half of the long side at worst
MIN_STD = 6.0  # a blank page or a flat colour (grey level std over the pixels) is rejected
HASH_SIDE = 8  # difference hash: (HASH_SIDE + 1) x HASH_SIDE greyscale thumbnail -> 64 bits

# stratum priority: paintings first (the scarce ones), the broad classes last; every object falls in exactly one stratum
STRATA = ("painting", "photograph", "drawing", "other_flat", "print", "object")
OTHER_FLAT = (
    "Watercolors",
    "Miniatures",
    "Posters",
    "Albums",
    "Paper",
    "Calligraphy",
    "Scrolls",
    "Prints-Posters",
)


@dataclass
class TrainsetConfig:
    """``configs/l2p/trainset.yaml``."""

    csv_url: str = (
        "https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv"
    )
    csv_sha256: str = ""
    target: int = (
        100_000  # images kept (the paper used 200,000 per data set; section 8.2 halves it)
    )
    seed: int = 29  # candidate order inside a stratum: random.Random(seed) over the sorted ids
    shard_size: int = 1000
    threads: int = (
        8  # parallel image downloads (images.metmuseum.org is a CDN, not the rate-limited API)
    )
    hash_distance: int = 6  # near-duplicate of an evaluation image: hamming distance of the 64-bit difference hashes (also h-flipped)
    val_images: int = 2000  # held-out images for the training curves (never trained on)
    eval_manifest: str = "configs/eval/eval_set.yaml"
    eval_prepared: str = "data/raw/eval/prepared"
    root: str = "data/raw/l2p_met"
    csv_name: str = "metadata_images.csv.gz"
    ids_file: str = "configs/l2p/trainset_ids.txt"
    excluded_file: str = "configs/l2p/trainset_excluded.yaml"
    strata_cap: dict[str, int] = field(default_factory=dict)  # optional per-stratum cap


def stratum(classification: str, department: str) -> str:
    """The stratum of a Met object from its first classification (``a|b`` keeps ``a``)."""
    c = classification.split("|")[0].strip()
    if c.startswith("Paintings"):
        return "painting"
    if c.startswith("Photographs"):
        return "photograph"
    if c.startswith("Drawings"):
        return "drawing"
    if c.startswith("Prints"):
        return "print"
    if c in OTHER_FLAT:
        return "other_flat"
    return "object"


def read_candidates(csv_path: Path) -> Iterator[dict[str, Any]]:
    """Public-domain objects with an image of ``metadata_images.csv.gz``: id, stratum, artist names, image URL and descriptive fields."""
    csv.field_size_limit(10**9)
    with gzip.open(csv_path, "rt", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["Is Public Domain"] != "True" or not row["primaryImageSmall"]:
                continue
            yield {
                "id": int(row["Object ID"]),
                "stratum": stratum(row["Classification"], row["Department"]),
                "artists": artist_names(row["Artist Display Name"]),
                "artist": row["Artist Display Name"][:200],
                "number": row["Object Number"],
                "url": row["primaryImageSmall"],
                "title": row["Title"][:200],
                "date": row["Object Date"],
                "classification": row["Classification"],
                "department": row["Department"],
            }


def artist_names(display: str) -> list[str]:
    """Normalized artist names of a Met ``Artist Display Name`` (``a|b`` = several); blank and anonymous names are dropped."""
    out = []
    for name in display.split("|"):
        n = " ".join(name.lower().split())
        if n and n not in ("unknown", "anonymous", "artist unknown", "unidentified artist"):
            out.append(n)
    return out


def exclusions(eval_manifest: Path) -> tuple[set[int], set[str]]:
    """Object ids and artist names of the Met images of the evaluation manifest (every dev and test image)."""
    data = yaml.safe_load(eval_manifest.read_text(encoding="utf-8"))
    ids: set[int] = set()
    artists: set[str] = set()
    for im in data["images"]:
        if im.get("source") == "met":
            ids.add(int(str(im["id"]).split("_")[1]))
            artists.update(artist_names(str(im.get("author") or "")))
    return ids, artists


def plan(
    candidates: Sequence[dict[str, Any]], cfg: TrainsetConfig, ids: set[int], artists: set[str]
) -> list[dict[str, Any]]:
    """Candidates that survive the exclusions, in fetch order: strata in :data:`STRATA` order, inside a stratum the sorted ids shuffled by ``random.Random(seed)``."""
    keep = [c for c in candidates if c["id"] not in ids and not (set(c["artists"]) & artists)]
    out: list[dict[str, Any]] = []
    for s in STRATA:
        pool = sorted((c for c in keep if c["stratum"] == s), key=lambda c: c["id"])
        random.Random(cfg.seed).shuffle(pool)
        out.extend(pool[: cfg.strata_cap.get(s, len(pool))])
    return out


def crop_resize(data: bytes) -> tuple[np.ndarray | None, str]:
    """Decode an image file, centre-crop to a square, area-resize to ``SIZE``: ``(uint8 (SIZE, SIZE, 3), "")`` or ``(None, reason)``."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:  # noqa: BLE001 - any decoding failure is a rejection with a reason
        return None, "undecodable"
    w, h = im.size
    if min(w, h) < MIN_SHORT_SIDE:
        return None, "small"
    if not MIN_ASPECT <= w / h <= MAX_ASPECT:
        return None, "aspect"
    rgb = im.convert("RGB")
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    arr = np.asarray(
        rgb.crop((left, top, left + s, top + s)).resize((SIZE, SIZE), Image.Resampling.BOX)
    )
    if float(arr.mean(2).std()) < MIN_STD:
        return None, "flat"
    return arr, ""


def dhash(arr: np.ndarray) -> int:
    """64-bit difference hash of a ``(H, W, 3)`` uint8 image: greyscale ``(HASH_SIDE + 1) x HASH_SIDE`` thumbnail, sign of the horizontal gradient."""
    g = Image.fromarray(arr).convert("L").resize((HASH_SIDE + 1, HASH_SIDE), Image.Resampling.BOX)
    t = np.asarray(g, dtype=np.int16)
    bits = (t[:, 1:] > t[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def eval_hashes(manifest: Path, prepared: Path) -> dict[str, tuple[int, int]]:
    """``(hash, hash of the h-flip)`` of every prepared evaluation image at the training size (box filter, so it matches ``crop_resize``)."""
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    out: dict[str, tuple[int, int]] = {}
    for im in data["images"]:
        p = prepared / f"{im['id']}.png"
        if not p.exists():
            continue
        arr = np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BOX))
        out[str(im["id"])] = (dhash(arr), dhash(arr[:, ::-1].copy()))
    return out


def near_duplicates(
    hashes: dict[int, int], reference: dict[str, tuple[int, int]], distance: int
) -> dict[int, str]:
    """Object ids whose hash is within ``distance`` bits of an evaluation image (or its mirror): ``{object id: evaluation id}``."""
    ref = [(k, h) for k, pair in reference.items() for h in pair]
    out: dict[int, str] = {}
    for oid, h in hashes.items():
        for k, r in ref:
            if hamming(h, r) <= distance:
                out[oid] = k
                break
    return out


def pixel_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def manifest_digest(rows: Sequence[tuple[int, str]]) -> str:
    """SHA-256 of the sorted ``id:pixel_sha256`` lines (the one value that pins the whole subset)."""
    text = "\n".join(f"{i}:{h}" for i, h in sorted(rows))
    return hashlib.sha256(text.encode()).hexdigest()


def class_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(r["stratum"] for r in rows))
