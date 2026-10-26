"""The evaluation image set (T-024): manifest schema, validation, preparation and the seeded draws.

Spec: ``docs/design/evaluation.md`` section 1. The set is 100 images (40 paintings, 30 photographs,
30 SDXL images); the manifest ``configs/eval/eval_set.yaml`` lists every image with its source,
its licence,
the SHA-256 of the file used and the SHA-256 of the prepared 512 x 512 pixels. Source files are
fetched, never committed.

Preparation (section 1.3): the source must have a short side of at least 768 px and an aspect
ratio in
[3:4, 4:3]; it is centre-cropped to a square (or to the hand-set ``crop_box`` of the manifest,
set before any
method run) and reduced to 512 x 512 with a box (area) filter. The box filter of Pillow averages
the source pixels
covered by each output pixel, so the result does not depend on the torch version, which keeps
the hashes stable.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

SIZE = 512  # side of the prepared images
MIN_SHORT_SIDE = 768  # a source is eligible when its short side is at least this
MIN_ASPECT, MAX_ASPECT = 3 / 4, 4 / 3  # width / height of an eligible source
DRAW_SEED = 22  # ``random.Random(22)``: the stratified draws and the dev split

# category -> stratum -> number of images (evaluation.md section 1.1)
QUOTAS: dict[str, dict[str, int]] = {
    "painting": {
        "european_portrait": 8,
        "european_landscape": 8,
        "european_other": 8,
        "asian": 8,
        "other_art": 8,
    },
    "photo": {"photo_portrait": 10, "photo_animal": 7, "photo_object": 6, "photo_scene": 7},
    "sdxl": {"sdxl_portrait": 10, "sdxl_scene": 10, "sdxl_object": 10},
}
# dev split (section 1.1): images per stratum group, the SDXL ones are fixed by prompt index
DEV_SDXL_PROMPTS = (4, 14, 25)
DEV_PAINTINGS = {"european": 2, "asian": 1, "other_art": 1}
DEV_PHOTOS = {"photo_portrait": 1, "photo_other": 2}
LICENCES = (  # accepted licence strings (section 1.2): CC0 / public domain, or own-generated
    "CC0 1.0 (Met Open Access, isPublicDomain)",
    "Public domain (US government work, Library of Congress FSA/OWI)",
    "Public domain (US government work, US Fish and Wildlife Service)",
    "Public domain (Wikimedia Commons licence tag)",
    "CC0 1.0 (own-generated, SDXL output)",
)
MAX_GREYSCALE_PHOTOS = 10
MIN_COLOUR_PHOTOS = 20


@dataclass
class EvalImage:
    """One manifest entry. ``sha256`` is that of the downloaded file (for SDXL, of the 1024 px PNG
    pixels)."""

    id: str
    category: str
    stratum: str
    source: str
    dev: bool
    licence: str
    retrieved: str
    sha256: str
    prepared_sha256: str
    title: str = ""
    author: str = ""
    date: str = ""
    page_url: str = ""
    url: str = ""
    source_id: str = ""
    subtype: str = ""
    width: int = 0
    height: int = 0
    colour: bool = True
    crop_box: list[int] | None = None  # (left, top, right, bottom) in source pixels, if hand-set
    prompt: str = ""
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_sha256(pixels: np.ndarray) -> str:
    """SHA-256 of the raw ``uint8`` RGB pixels (shape ``(H, W, 3)``, C order), independent of the
    PNG encoder."""
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"expected a uint8 (H, W, 3) array, got {pixels.dtype} {pixels.shape}")
    return sha256_bytes(np.ascontiguousarray(pixels).tobytes())


def eligible(width: int, height: int) -> bool:
    """True if the source may be prepared without padding (section 1.3)."""
    return min(width, height) >= MIN_SHORT_SIDE and MIN_ASPECT <= width / height <= MAX_ASPECT


def prepare_image(
    im: Image.Image, crop_box: Sequence[int] | None = None, size: int = SIZE
) -> np.ndarray:
    """Centre-crop to a square (or ``crop_box``), reduce to ``size`` x ``size`` with the box filter.

    Returns ``uint8`` RGB ``(size, size, 3)``. Greyscale and palette images are converted to
    three equal channels /
    RGB; an alpha channel is dropped. Raises ``ValueError`` for an ineligible source without a
    ``crop_box``.
    """
    rgb = im.convert("RGB")
    w, h = rgb.size
    if crop_box is None:
        if not eligible(w, h):
            raise ValueError(
                f"ineligible source {w}x{h} (short side >= {MIN_SHORT_SIDE}, aspect in [3:4, 4:3])"
            )
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        box = (left, top, left + side, top + side)
    else:
        box = (int(crop_box[0]), int(crop_box[1]), int(crop_box[2]), int(crop_box[3]))
        if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            raise ValueError(f"crop_box {box} outside the {w}x{h} source")
        if box[2] - box[0] != box[3] - box[1]:
            raise ValueError(f"crop_box {box} is not square")
    return np.asarray(rgb.resize((size, size), Image.Resampling.BOX, box=box), dtype=np.uint8)


def is_colour(pixels: np.ndarray, threshold: float = 8.0) -> bool:
    """A photograph is in colour if the mean channel spread (max - min over RGB, 0..255) exceeds
    ``threshold``."""
    spread = pixels.max(axis=2).astype(np.float32) - pixels.min(axis=2).astype(np.float32)
    return bool(spread.mean() > threshold)


def seeded_order(ids: Sequence[Any], rng: random.Random) -> list[Any]:
    """The draw order of a stratum: the ids sorted, then shuffled by ``rng`` (the first acceptable
    ones are kept)."""
    order = sorted(ids)
    rng.shuffle(order)
    return order


def draw_dev_split(images: Sequence[EvalImage], rng: random.Random) -> set[str]:
    """Ids of the dev images (section 1.1): 4 paintings (2 European, 1 Asian, 1 other), 3
    photographs (1 portrait,
    2 other), and the SDXL prompts 04, 14, 25. Drawn inside each group over the ids sorted, with
    ``rng``."""
    dev: set[str] = set()

    def pick(pool: list[EvalImage], n: int) -> None:
        ids = sorted(i.id for i in pool)
        rng.shuffle(ids)
        dev.update(ids[:n])

    paintings = [i for i in images if i.category == "painting"]
    pick([i for i in paintings if i.stratum.startswith("european")], DEV_PAINTINGS["european"])
    pick([i for i in paintings if i.stratum == "asian"], DEV_PAINTINGS["asian"])
    pick([i for i in paintings if i.stratum == "other_art"], DEV_PAINTINGS["other_art"])
    photos = [i for i in images if i.category == "photo"]
    pick([i for i in photos if i.stratum == "photo_portrait"], DEV_PHOTOS["photo_portrait"])
    pick([i for i in photos if i.stratum != "photo_portrait"], DEV_PHOTOS["photo_other"])
    dev.update(i.id for i in images if i.category == "sdxl" and i.seed in DEV_SDXL_PROMPTS)
    return dev


def load_manifest(path: Path) -> tuple[dict[str, Any], list[EvalImage]]:
    """Read the manifest: (header, images). Unknown keys of an image go to ``extra``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    known = set(EvalImage.__dataclass_fields__) - {"extra"}
    images = []
    for entry in raw["images"]:
        fields = {k: v for k, v in entry.items() if k in known}
        fields["extra"] = {k: v for k, v in entry.items() if k not in known}
        images.append(EvalImage(**fields))
    return {k: v for k, v in raw.items() if k != "images"}, images


def dump_manifest(path: Path, header: dict[str, Any], images: Sequence[EvalImage]) -> None:
    entries = []
    for im in images:
        d = {k: getattr(im, k) for k in EvalImage.__dataclass_fields__ if k != "extra"}
        keep = {"id", "category", "stratum", "source", "dev", "licence", "retrieved"}
        keep |= {"sha256", "prepared_sha256"}
        d = {k: v for k, v in d.items() if v not in ("", None) or k in keep}
        d.update(im.extra)
        entries.append(d)
    body = yaml.safe_dump(
        {**header, "images": entries}, sort_keys=False, allow_unicode=True, width=120
    )
    path.write_text(body, encoding="utf-8", newline="\n")


def validate_manifest(header: dict[str, Any], images: Sequence[EvalImage]) -> list[str]:
    """The problems of a manifest (empty list = valid): counts per category and stratum, the dev
    split, unique ids,
    licence strings, checksums, the colour rule of the photographs. Checks the structure, not
    the files."""
    problems: list[str] = []
    ids = [i.id for i in images]
    if len(set(ids)) != len(ids):
        problems.append(f"duplicate ids: {sorted(k for k, v in Counter(ids).items() if v > 1)}")
    by_stratum = Counter((i.category, i.stratum) for i in images)
    for cat, strata in QUOTAS.items():
        for stratum, n in strata.items():
            if by_stratum.get((cat, stratum), 0) != n:
                problems.append(
                    f"{cat}/{stratum}: {by_stratum.get((cat, stratum), 0)} images, expected {n}"
                )
    unknown = set(by_stratum) - {(c, s) for c, st in QUOTAS.items() for s in st}
    if unknown:
        problems.append(f"unknown strata: {sorted(unknown)}")
    dev = [i for i in images if i.dev]
    expected_dev = {"painting": 4, "photo": 3, "sdxl": 3}
    for cat, n in expected_dev.items():
        got = sum(1 for i in dev if i.category == cat)
        if got != n:
            problems.append(f"dev {cat}: {got} images, expected {n}")
    sdxl_dev = sorted(i.seed for i in dev if i.category == "sdxl" and i.seed is not None)
    if sdxl_dev != sorted(DEV_SDXL_PROMPTS):
        problems.append(f"dev SDXL prompts {sdxl_dev}, expected {sorted(DEV_SDXL_PROMPTS)}")
    for im in images:
        if im.licence not in LICENCES:
            problems.append(f"{im.id}: licence {im.licence!r} is not an accepted string")
        if len(im.sha256) != 64 or len(im.prepared_sha256) != 64:
            problems.append(f"{im.id}: missing checksum")
        if not im.retrieved:
            problems.append(f"{im.id}: missing retrieval date")
        if im.category != "sdxl" and not (im.url and im.page_url):
            problems.append(f"{im.id}: missing source URL")
        if im.category == "sdxl" and (not im.prompt or im.seed is None):
            problems.append(f"{im.id}: SDXL entry without prompt or seed")
    photos = [i for i in images if i.category == "photo"]
    grey = sum(1 for i in photos if not i.colour)
    if grey > MAX_GREYSCALE_PHOTOS or len(photos) - grey < MIN_COLOUR_PHOTOS:
        problems.append(
            f"{grey} greyscale photographs of {len(photos)} "
            f"(max {MAX_GREYSCALE_PHOTOS}, at least {MIN_COLOUR_PHOTOS} in colour)"
        )
    if header.get("draw_seed") != DRAW_SEED:
        problems.append(f"draw_seed {header.get('draw_seed')} != {DRAW_SEED}")
    return problems


# ----------------------------------------------------------------------------- borders (one
# definition, used to trim and to report)

BORDER_MAX_AREA = 0.05  # protocol 1.2.3: no frame or foreign element over 5 % of the area
FLAT_LUMA, FLAT_STD = (
    28.0,
    6.0,
)  # "flat" border line: near black and uniform (paintings, prints, mats)
FILM_LUMA = (
    30.0  # "film" border line (scanned slides): mean luminance below this, frame numbers included
)
BORDER_SKIP = 3  # thin light edge lines (px at 512) skipped before the band


def border_flags(grey: np.ndarray, kind: str) -> np.ndarray:
    """True for each row of ``grey`` (H, W; 0-255) that is a border line of the given ``kind``.

    ``flat``: a near-black, uniform line (mean < 28 and std < 6), so a dark but textured painting is
    never a border. ``film``: any line whose mean luminance is below 30 (the black film edge of a
    scanned slide, with its frame numbers and soft inner edge). Pass ``grey.T`` for the columns."""
    if kind == "film":
        return np.asarray(grey.mean(axis=1) < FILM_LUMA)
    if kind == "flat":
        return np.asarray((grey.mean(axis=1) < FLAT_LUMA) & (grey.std(axis=1) < FLAT_STD))
    raise ValueError(f"border kind {kind!r}: expected 'flat' or 'film'")


def edge_band(flags: np.ndarray, limit: int, max_skip: int = BORDER_SKIP) -> int:
    """Width of the border band at the start of ``flags`` (True = border line).

    The thin light edge lines in front of the band are included: up to ``max_skip`` light lines are
    left out (a scanned slide often has a 1-2 px bright line before its black film edge), then the
    consecutive border lines are counted, at most ``limit`` of them."""
    start = next((k for k in range(min(max_skip + 1, len(flags))) if flags[k]), None)
    if start is None:
        return 0
    n = start
    while n < limit and n < len(flags) and flags[n]:
        n += 1
    return n


def border_widths(grey: np.ndarray, kind: str, limit: float = 0.25) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) widths of the border bands of ``grey`` (at most ``limit`` of
    each side). The light-line
    tolerance is 3 px at 512 px and scales with the image."""
    h, w = grey.shape
    skip = max(BORDER_SKIP, round(BORDER_SKIP * min(h, w) / SIZE))
    rows, cols = border_flags(grey, kind), border_flags(grey.T, kind)
    return (
        edge_band(rows, int(h * limit), skip),
        edge_band(rows[::-1], int(h * limit), skip),
        edge_band(cols, int(w * limit), skip),
        edge_band(cols[::-1], int(w * limit), skip),
    )


def grey_border_area(grey: np.ndarray, kind: str) -> float:
    """Fraction of the area of a greyscale image (H, W) taken by its border bands."""
    top, bottom, left, right = border_widths(grey, kind, limit=0.5)
    h, w = grey.shape
    return float((top * w + bottom * w + (left + right) * (h - top - bottom)) / (h * w))


def border_area_fraction(pixels: np.ndarray, kind: str) -> float:
    """Border fraction of an RGB image (the figure reported against the 5 % rule)."""
    return grey_border_area(
        np.asarray(Image.fromarray(pixels).convert("L"), dtype=np.float32), kind
    )


def border_crop_box(
    im: Image.Image, min_side: int = 640, kind: str = "flat"
) -> tuple[bool, list[int] | None]:
    """Square crop box (source pixels) that leaves out the border bands (same definition as the
    report), or ``None``
    for the plain centre crop. ``ok`` is False if the trimmed square is under ``min_side``
    pixels."""
    grey = np.asarray(im.convert("L"), dtype=np.float32)
    h, w = grey.shape
    top, bottom, left, right = border_widths(grey, kind)
    if top + bottom + left + right == 0:
        return True, None
    tw, th = w - left - right, h - top - bottom
    side = min(tw, th)
    if side < min_side:
        return False, None
    x0, y0 = left + (tw - side) // 2, top + (th - side) // 2
    return True, [x0, y0, x0 + side, y0 + side]
