"""Audit of the evaluation set against every machine-checkable rule of the protocol (T-034).

    python -m scripts.audit_eval_set [--root data/raw/eval] [--no-detector]

Run after ``python -m scripts.fetch_eval_set``. Every prepared image is checked; the report is written to
``docs/experiments/data/eval_set/audit.json`` and the exit status is 1 if any rule has a violation. Rules:

1. manifest: counts per category and stratum, dev composition, unique ids, licence strings, checksums, retrieval dates;
2. files: the prepared pixels and the SDXL files match their SHA-256; every ``crop_box`` is inside its source and square;
3. borders (protocol 1.2.3, at most 5 % of the area): one definition per kind of source, the one used to trim
   (``flat`` bars for paintings and prints, ``film`` edges for scanned slides), plus the ``film`` definition applied to
   every photograph as an independent check;
4. near-duplicates: all pairs of prepared images (32 x 32 correlation), duplicated sources, same-scene pairs of the same
   photographer, and dev images that look like test images (stricter threshold);
5. photographers: at most ``MAX_PER_PHOTOGRAPHER`` images per normalized name;
6. dev/test: disjoint by construction and by content, SDXL dev prompts fixed;
7. sources: URL hosts on the allow-list, clean dates, no control character in any text file of the repository or in the
   manifest strings, children excluded from the Commons titles and categories, the people detector finds nobody in the
   modern agency photographs;
8. colour: a photograph flagged in colour is not a greyscale or a one-hue image.

The rules that need eyes (clear subject for the stratum, frame or foreign element, possibly living identifiable person,
text overlay) are recorded image by image in ``docs/experiments/data/eval_set/visual_audit.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from brushstroke.eval.dataset import (
    BORDER_MAX_AREA,
    EvalImage,
    border_area_fraction,
    grey_border_area,
    is_colour,
    load_manifest,
    pixel_sha256,
    validate_manifest,
)
from scripts.build_eval_set import (
    CHILD,
    FWS,
    LOC_FSA,
    MANIFEST,
    MAX_PER_PHOTOGRAPHER,
    frequent_categories,
    has_person,
    photographers,
    same_series,
    thumb,
)

REPORT = Path("docs/experiments/data/eval_set/audit.json")
HOSTS = {"images.metmuseum.org", "upload.wikimedia.org"}
DUPLICATE_CORRELATION = 0.9  # 32 x 32 greyscale correlation between two set images
DEV_TEST_CORRELATION = 0.75  # a dev image that resembles a test image this much is a leak risk
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".cfg"}
SKIP_DIRS = {".venv", "outputs", "data", ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache"}


def border_kind(im: EvalImage) -> str:
    return "film" if im.licence == LOC_FSA else "flat"


def load_prepared(root: Path, im: EvalImage) -> np.ndarray:
    return np.asarray(Image.open(root / "prepared" / f"{im.id}.png").convert("RGB"))


def check_files(root: Path, images: list[EvalImage]) -> list[str]:
    bad: list[str] = []
    for im in images:
        path = root / "prepared" / f"{im.id}.png"
        if not path.exists():
            bad.append(f"{im.id}: prepared image missing (run scripts.fetch_eval_set)")
            continue
        if pixel_sha256(load_prepared(root, im)) != im.prepared_sha256:
            bad.append(f"{im.id}: prepared pixels differ from the manifest")
        if im.category == "sdxl":
            raw = root / "sdxl" / f"{im.id}.png"
            if (
                not raw.exists()
                or pixel_sha256(np.asarray(Image.open(raw).convert("RGB"))) != im.sha256
            ):
                bad.append(f"{im.id}: SDXL file missing or different from the manifest")
            continue
        src = next((root / "source").glob(f"{im.id}.*"), None)
        if src is None:
            bad.append(f"{im.id}: source file missing")
            continue
        with Image.open(src) as pil:
            w, h = pil.size
        want = str(im.extra.get("commons_sha1", ""))
        if want and hashlib.sha1(src.read_bytes()).hexdigest() != want:  # noqa: S324
            bad.append(
                f"{im.id}: the source file is not the file the Commons API describes (SHA-1)"
            )
        if (w, h) != (im.width, im.height):
            bad.append(f"{im.id}: source is {w}x{h}, manifest says {im.width}x{im.height}")
        if im.crop_box is not None:
            x0, y0, x1, y1 = im.crop_box
            if not (
                0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and x1 - x0 == y1 - y0 and x1 - x0 >= 640
            ):
                bad.append(
                    f"{im.id}: crop_box {im.crop_box} is not a square of at least 640 px inside {w}x{h}"
                )
    return bad


def source_crop_grey(root: Path, im: EvalImage) -> np.ndarray:
    """The part of the source that becomes the prepared image (``crop_box`` or the centre square), full resolution."""
    src = next((root / "source").glob(f"{im.id}.*"))
    with Image.open(src) as pil:
        grey = np.asarray(pil.convert("L"), dtype=np.float32)
    if im.crop_box is not None:
        x0, y0, x1, y1 = im.crop_box
        return grey[y0:y1, x0:x1]
    h, w = grey.shape
    side = min(h, w)
    return grey[(h - side) // 2 : (h - side) // 2 + side, (w - side) // 2 : (w - side) // 2 + side]


def check_borders(root: Path, images: list[EvalImage]) -> tuple[dict[str, float], list[str]]:
    """Border bands against the 5 % rule: on the cropped source (the definition used to trim, at full resolution), and
    for the scanned slides also on the 512 px prepared image (an independent scale)."""
    fractions: dict[str, float] = {}
    bad: list[str] = []
    for im in images:
        if im.category == "sdxl":
            continue
        kind = border_kind(im)
        own = grey_border_area(source_crop_grey(root, im), kind)
        fractions[im.id] = round(own, 4)
        if own > BORDER_MAX_AREA:
            bad.append(f"{im.id}: border {own:.1%} of the cropped source ({kind} definition)")
        if kind == "film":
            small = border_area_fraction(load_prepared(root, im), "film")
            fractions[im.id + "@512"] = round(small, 4)
            if small > BORDER_MAX_AREA:
                bad.append(f"{im.id}: film-definition border {small:.1%} of the prepared image")
    return fractions, bad


def check_duplicates(root: Path, images: list[EvalImage]) -> tuple[float, list[str]]:
    thumbs = {im.id: thumb(load_prepared(root, im)) for im in images}
    ids = [im.id for im in images]
    by_id = {im.id: im for im in images}
    bad: list[str] = []
    worst = 0.0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            c = float((thumbs[ids[a]] * thumbs[ids[b]]).mean())
            worst = max(worst, c)
            ia, ib = by_id[ids[a]], by_id[ids[b]]
            if c > DUPLICATE_CORRELATION:
                bad.append(f"{ia.id} and {ib.id}: thumbnails correlate at {c:.2f}")
            elif ia.dev != ib.dev and c > DEV_TEST_CORRELATION:
                bad.append(f"dev/test: {ia.id} and {ib.id} correlate at {c:.2f}")
    keys = Counter((im.source, im.source_id) for im in images if im.source_id)
    bad += [f"source {k} appears {n} times" for k, n in keys.items() if n > 1]
    titles = Counter((im.author, im.title) for im in images if im.category == "painting")
    bad += [f"painting {k} appears {n} times" for k, n in titles.items() if n > 1]
    commons = [
        {
            "id": im.id,
            "photographers": photographers(im.author),
            "categories": set(str(im.extra.get("commons_categories", "")).split("|")),
            "title": im.source_id,
            "agency": im.licence == FWS,
        }
        for im in images
        if im.source == "commons"
    ]
    frequent = frequent_categories(commons)
    for k, a in enumerate(commons):
        for b in commons[k + 1 :]:
            if same_series(a, b, frequent):
                bad.append(f"same scene or series: {a['id']} and {b['id']}")
    return round(worst, 3), bad


def check_photographers(images: list[EvalImage]) -> tuple[dict[str, int], list[str]]:
    count: Counter[str] = Counter()
    for im in images:
        if im.source == "commons":
            count.update(photographers(im.author))
    bad = [
        f"photographer {n!r}: {c} images (max {MAX_PER_PHOTOGRAPHER})"
        for n, c in count.items()
        if c > MAX_PER_PHOTOGRAPHER
    ]
    return dict(count.most_common()), bad


def check_dev_test(images: list[EvalImage]) -> list[str]:
    bad: list[str] = []
    dev = {im.id for im in images if im.dev}
    test = {im.id for im in images if not im.dev}
    if dev & test or len(dev) != 10 or len(test) != 90:
        bad.append(f"dev {len(dev)} / test {len(test)} images, expected 10 / 90 disjoint")
    for im in images:
        if im.category == "sdxl" and im.dev != (im.seed in (4, 14, 25)):
            bad.append(f"{im.id}: dev flag does not follow the fixed SDXL prompts 04, 14, 25")
    return bad


def check_sources(images: list[EvalImage]) -> list[str]:
    bad: list[str] = []
    for im in images:
        if im.category != "sdxl":
            host = urllib.parse.urlparse(im.url).netloc
            if host not in HOSTS:
                bad.append(f"{im.id}: host {host!r} is not on the allow-list")
            if not im.page_url.startswith("https://"):
                bad.append(f"{im.id}: page URL is not https")
        for field in ("title", "author", "date", "url", "page_url", "source_id", "prompt"):
            value = str(getattr(im, field))
            if any(ord(c) < 32 for c in value) or "QS:P" in value or value.startswith("Taken on"):
                bad.append(f"{im.id}: unclean {field} {value[:60]!r}")
        if im.source == "commons" and im.licence != FWS:
            text = f"{im.source_id} {im.title} {im.extra.get('commons_categories', '')}"
            if CHILD.search(text.replace("_", " ")):
                bad.append(f"{im.id}: a child word in the title or categories")
    return bad


def check_control_characters() -> list[str]:
    bad = []
    for path in Path(".").rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES or set(path.parts) & SKIP_DIRS:
            continue
        if any(b < 32 and b not in (9, 10, 12, 13) for b in path.read_bytes()):
            bad.append(f"{path}: control character")
    return bad


def check_colour(root: Path, images: list[EvalImage]) -> list[str]:
    bad = []
    for im in images:
        if im.category == "photo" and im.colour and not is_colour(load_prepared(root, im)):
            bad.append(f"{im.id}: flagged in colour but greyscale")
    return bad


def check_people(
    root: Path, images: list[EvalImage], use_detector: bool
) -> tuple[list[str], list[str]]:
    agency = [im for im in images if im.licence == FWS]
    found = [im.id for im in agency if use_detector and has_person(load_prepared(root, im))]
    return [f"{i}: a person is detected in an agency photograph" for i in found], [
        im.id for im in agency
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("data/raw/eval"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--no-detector",
        action="store_true",
        help="skip the people detector (no network or weights)",
    )
    args = parser.parse_args(argv)
    header, images = load_manifest(args.manifest)
    results: dict[str, Any] = {}
    violations: dict[str, list[str]] = {}

    violations["manifest"] = validate_manifest(header, images)
    violations["files"] = check_files(args.root, images)
    if violations["files"]:  # nothing else is meaningful without the files
        results["violations"] = violations
    else:
        results["border_area"], violations["borders"] = check_borders(args.root, images)
        results["max_thumbnail_correlation"], violations["duplicates"] = check_duplicates(
            args.root, images
        )
        results["photographers"], violations["photographers"] = check_photographers(images)
        violations["dev_test"] = check_dev_test(images)
        violations["sources"] = check_sources(images)
        violations["control_characters"] = check_control_characters()
        violations["colour"] = check_colour(args.root, images)
        violations["people"], results["agency_photographs"] = check_people(
            args.root, images, not args.no_detector
        )
        results["violations"] = violations
    results["images"] = len(images)
    results["strata"] = dict(Counter(f"{im.category}/{im.stratum}" for im in images))
    results["max_border_area"] = max(results.get("border_area", {"-": 0.0}).values())
    total = sum(len(v) for v in violations.values())
    results["total_violations"] = total
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    for rule, items in violations.items():
        print(f"{rule:20s} {len(items)} violation(s)")
        for item in items[:20]:
            print("   ", item)
    print(f"total: {total} violation(s) in {len(images)} images; report: {REPORT}")
    if total:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
