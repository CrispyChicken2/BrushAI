"""Curation of the evaluation set (T-024): the seeded stratified draw that produces ``configs/eval/eval_set.yaml``.

    python -m scripts.build_eval_set curate [--insecure]   # draw, download, check, write the manifest (needs the network)
    python -m scripts.build_eval_set sheets                # contact sheets in outputs/eval_set/ for the visual check

This is the one-off curation, not the reproduction path: the manifest it writes is the frozen result, and
``python -m scripts.fetch_eval_set`` reproduces the images from it. The draw follows ``docs/design/evaluation.md``
section 1: per stratum, the candidate ids sorted and shuffled by ``random.Random(22)``, the first candidates that pass the
rules are kept. Rules (all mechanical, none looks at a method result):

* Met Open Access paintings: ``isPublicDomain``, a primary image, classification "Paintings" (Asian art: paintings, and
  woodblock prints), eligible size and aspect (short side >= 768, aspect in [3:4, 4:3]); strata from the Met subject tags.
* Photographs from Wikimedia Commons: the file page must carry a public-domain licence tag and a US-government source
  (Library of Congress FSA/OWI colour slides, US Fish and Wildlife Service, Library of Congress Prokudin-Gorskii); JPEG,
  eligible size, in colour, at most 3 images per photographer; people only adults of the 1900s to 1940s (no child, no
  living person is identifiable); strata from the file categories and title.
* Visual rejections (watermark / text / frame over 5 % of the area, no clear subject, face not salient) are listed with a
  reason in ``configs/eval/curation_rejections.yaml``; they are decided on the contact sheets before any method runs.
* Nothing equal or near-equal to an image used for tuning so far (T-009 sweep, demos) is accepted.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from brushstroke.eval.dataset import (
    DRAW_SEED,
    QUOTAS,
    EvalImage,
    dump_manifest,
    eligible,
    is_colour,
    load_manifest,
    pixel_sha256,
    prepare_image,
    seeded_order,
    sha256_bytes,
)
from brushstroke.utils.http import Http

MET = "https://collectionapi.metmuseum.org/public/collection/v1"
COMMONS = "https://commons.wikimedia.org/w/api.php"
RAW = Path("data/raw/eval")
MANIFEST = Path("configs/eval/eval_set.yaml")
REJECTIONS = Path("configs/eval/curation_rejections.yaml")
SDXL_PROMPTS = Path("configs/eval/sdxl_prompts.yaml")
TUNING = [  # the images used for tuning so far (T-009 sweep, M1/M2 demos): never in the set
    "earth_apollo17", "great_wave", "kirkjufell", "mona_lisa", "pearl_earring", "starry_night",
]  # fmt: skip
MET_EXCLUDED = {45434}  # The Great Wave off Kanagawa (Met impression), in the T-009 sweep
TODAY = time.strftime("%Y-%m-%d")
MET_LICENCE = "CC0 1.0 (Met Open Access, isPublicDomain)"
LOC_FSA = "Public domain (US government work, Library of Congress FSA/OWI)"
FWS = "Public domain (US government work, US Fish and Wildlife Service)"
COMMONS_PD = "Public domain (Wikimedia Commons licence tag)"
MAX_PER_PHOTOGRAPHER = 3
MAX_FILE_BYTES = 40_000_000

PORTRAIT_TAGS = {"Portraits"}
LANDSCAPE_TAGS = {"Landscapes", "Cityscapes", "Seascapes", "Rivers", "Mountains", "Marine painting"}
PEOPLE = re.compile(
    r"\b(portrait|mr\.|mrs\.|miss|farmer|farm woman|worker|man|woman|men|women|wife|husband|mother|father|"
    r"cowboy|sheriff|blacksmith|preacher|miner|operator|conductor|owner|merchant|rancher)\b",
    re.I,
)
CHILD = re.compile(
    r"\b(child|children|boy|boys|girl|girls|baby|babies|infant|kid|kids|school|student|students|youth|teen|"
    r"teenage|son|daughter|family|scouts|pupil|pupils|4-h)\b",
    re.I,
)
ANIMAL = re.compile(
    r"\b(horse|horses|mule|mules|cow|cows|cattle|bull|steer|sheep|lamb|pig|pigs|hog|hogs|dog|dogs|chicken|"
    r"chickens|rooster|hen|goat|turkey|bird|birds|eagle|owl|deer|elk|bear|wolf|fox|seal|duck|goose|geese|swan|"
    r"heron|cat|moose|bison|buffalo|pelican|fawn|calf)\b",
    re.I,
)
OBJECT = re.compile(
    r"\b(truck|tractor|automobile|car|sign|machine|plow|barn|wagon|bridge|locomotive|train|boat|ship|lantern|"
    r"bottle|store|church|mill|pump|tool|windmill|silo|tent|trailer|bicycle|gate|fence|shed|shack|cabin)\b",
    re.I,
)
SCENE = re.compile(
    r"\b(landscape|street|field|fields|farm|village|town|harbor|harbour|river|lake|mountain|valley|coast|"
    r"mine|camp|city|canal|refuge|forest|desert|prairie|hills|sunset|view|panorama)\b",
    re.I,
)


# ----------------------------------------------------------------------------- network


def cached_json(http: Http, url: str, name: str) -> Any:
    path = RAW / "cache" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = http.json(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def commons_api(http: Http, **params: Any) -> Any:
    return http.json(COMMONS + "?" + urllib.parse.urlencode({"format": "json", **params}))


def strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


# ----------------------------------------------------------------------------- tuning images and rejections


def thumb(pixels: np.ndarray, n: int = 32) -> np.ndarray:
    im = Image.fromarray(pixels).convert("L").resize((n, n), Image.Resampling.BOX)
    v = np.asarray(im, dtype=np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def tuning_thumbs() -> list[np.ndarray]:
    out = []
    for name in TUNING:
        path = Path("data/raw") / f"{name}.jpg"
        if path.exists():
            with Image.open(path) as im:
                w, h = im.size
                s = min(w, h)
                out.append(
                    thumb(
                        np.asarray(
                            im.convert("RGB").crop(
                                ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
                            )
                        )
                    )
                )
    return out


def near_tuning(pixels: np.ndarray, tuning: list[np.ndarray]) -> bool:
    t = thumb(pixels)
    return any(float((t * u).mean()) > 0.9 for u in tuning)


def border_crop_box(
    im: Image.Image, min_side: int = 640, lenient: bool = False
) -> tuple[bool, list[int] | None]:
    """Square crop box (source pixels) that leaves out uniform black borders (film-holder edges of scanned
    slides, mat bars), or ``None`` for the plain centre crop. A border row/column is near black (mean < 28) and
    flat (std < 6), scanned inward from each edge over at most a quarter of the side; dark but textured paintings
    are never trimmed (``lenient``, for scanned slides, also accepts a row where 90 % of the pixels are near black:
    the film edge carries frame numbers). Returns ``(ok, box)``; ``ok`` is False if the trimmed square is under ``min_side`` pixels."""
    grey = np.asarray(im.convert("L"), dtype=np.float32)
    h, w = grey.shape

    def is_border(line: np.ndarray) -> bool:
        if lenient:
            return bool((line < 40).mean() >= 0.9)
        return bool(line.mean() < 28 and line.std() < 6)

    def edge(lines: np.ndarray, limit: int) -> int:
        n = 0
        while n < limit and is_border(lines[n]):
            n += 1
        return n

    top, bottom = edge(grey, h // 4), edge(grey[::-1], h // 4)
    left, right = edge(grey.T, w // 4), edge(grey.T[::-1], w // 4)
    if top + bottom + left + right == 0:
        return True, None
    tw, th = w - left - right, h - top - bottom
    side = min(tw, th)
    if side < min_side:
        return False, None
    x0, y0 = left + (tw - side) // 2, top + (th - side) // 2
    return True, [x0, y0, x0 + side, y0 + side]


def load_rejections() -> dict[str, str]:
    if not REJECTIONS.exists():
        return {}
    return {
        str(k): str(v)
        for k, v in (yaml.safe_load(REJECTIONS.read_text(encoding="utf-8")) or {}).items()
    }


# ----------------------------------------------------------------------------- Met paintings


def met_stratum(meta: dict[str, Any], dept: str) -> tuple[str, str] | None:
    """(stratum, subtype) of a Met object, or None if it does not belong to any painting stratum."""
    if not meta.get("isPublicDomain") or not meta.get("primaryImage"):
        return None
    cls = meta.get("classification", "")
    if dept == "european":
        if cls != "Paintings":
            return None
        tags = {t["term"] for t in (meta.get("tags") or [])}
        if tags & PORTRAIT_TAGS:
            return "european_portrait", "portrait"
        if tags & LANDSCAPE_TAGS:
            return "european_landscape", "landscape"
        return ("european_other", "other") if tags else None
    if dept == "asian":
        medium = meta.get("medium", "").lower()
        if cls == "Paintings":
            return "asian", "painting"
        if cls == "Prints" and "woodblock" in medium:
            return "asian", "woodblock_print"
        return None
    if (
        dept == "photo"
    ):  # Met photographs: historical sitters (nineteenth century), monochrome prints
        tags = {t["term"] for t in (meta.get("tags") or [])}
        if cls == "Photographs" and tags & PORTRAIT_TAGS:
            return "photo_portrait", "portrait"
        return None
    if (
        dept == "american"
    ):  # the American Wing leaves the classification empty: objectName says "Painting"
        return ("other_art", "american") if meta.get("objectName") == "Painting" else None
    if dept == "modern":
        return ("other_art", "modern") if cls == "Paintings" else None
    return None


MET_PHOTO_PORTRAITS = (
    8  # of the 10 photographic portraits (monochrome: 8 of the at most 10 greyscale photographs)
)
MET_DEPTS = {"european": 11, "asian": 6, "modern": 21, "american": 1, "photo": 19}
MET_QUOTAS = {  # (dept, stratum, subtype) -> n ; subtype quotas keep the Asian and "other" strata balanced
    ("european", "european_portrait", "portrait"): 8,
    ("european", "european_landscape", "landscape"): 8,
    ("european", "european_other", "other"): 8,
    ("asian", "asian", "painting"): 4,
    ("asian", "asian", "woodblock_print"): 4,
    ("modern", "other_art", "modern"): 4,
    ("american", "other_art", "american"): 4,
    ("photo", "photo_portrait", "portrait"): MET_PHOTO_PORTRAITS,
}


def download_image(http: Http, url: str, dest: Path) -> bytes:
    if dest.exists():
        return dest.read_bytes()
    data = http.get(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return data


def draw_met(
    http: Http, rng: random.Random, tuning: list[np.ndarray], rejected: dict[str, str]
) -> list[EvalImage]:
    out: list[EvalImage] = []
    for dept, dept_id in MET_DEPTS.items():
        ids = cached_json(http, f"{MET}/objects?departmentIds={dept_id}", f"met_dept_{dept_id}")[
            "objectIDs"
        ]
        need = {k: n for k, n in MET_QUOTAS.items() if k[0] == dept}
        kept = dict.fromkeys(need, 0)
        for oid in seeded_order(ids, rng):
            if all(kept[k] >= need[k] for k in need):
                break
            if oid in MET_EXCLUDED or f"met_{oid}" in rejected:
                continue
            try:
                meta = cached_json(http, f"{MET}/objects/{oid}", f"met_{oid}")
            except urllib.error.HTTPError:
                continue
            found = met_stratum(meta, dept)
            if found is None or kept.get((dept, *found), need.get((dept, *found), 0)) >= need.get(
                (dept, *found), 0
            ):
                continue
            key = (dept, *found)
            if key not in need:
                continue
            small = meta.get("primaryImageSmall")
            if small:  # cheap aspect check before the large download
                w0, h0 = _size_of(http, small, RAW / "source" / f"met_{oid}_small.jpg")
                if not 0.75 <= w0 / h0 <= 4 / 3:
                    continue
            ext = Path(urllib.parse.urlparse(meta["primaryImage"]).path).suffix.lower() or ".jpg"
            data = download_image(http, meta["primaryImage"], RAW / "source" / f"met_{oid}{ext}")
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
                if not eligible(w, h):
                    continue
                ok, box = border_crop_box(im)
                if not ok:
                    continue
                pixels = prepare_image(im, box)
            if near_tuning(pixels, tuning):
                continue
            tuning.append(
                thumb(pixels)
            )  # never accept a near-duplicate of an image already in the set
            kept[key] += 1
            out.append(
                EvalImage(
                    id=f"met_{oid}",
                    category="photo" if found[0].startswith("photo") else "painting",
                    stratum=found[0],
                    subtype=found[1],
                    source="met",
                    dev=False,
                    licence=MET_LICENCE,
                    retrieved=TODAY,
                    sha256=sha256_bytes(data),
                    prepared_sha256=pixel_sha256(pixels),
                    title=meta.get("title", ""),
                    author=meta.get("artistDisplayName", "") or "unknown",
                    date=meta.get("objectDate", ""),
                    page_url=meta.get("objectURL", ""),
                    url=meta["primaryImage"],
                    source_id=str(oid),
                    width=w,
                    height=h,
                    colour=not found[0].startswith("photo"),
                    crop_box=box,
                    extra={
                        "classification": meta.get("classification", ""),
                        "medium": meta.get("medium", ""),
                        "department": meta.get("department", ""),
                    },
                )
            )
            print(
                f"  {dept}: {found} <- {oid} {meta.get('title', '')[:50]!r} ({sum(kept.values())}/{sum(need.values())})",
                flush=True,
            )
    return out


def _size_of(http: Http, url: str, dest: Path) -> tuple[int, int]:
    with Image.open(io.BytesIO(download_image(http, url, dest))) as im:
        return im.size


# ----------------------------------------------------------------------------- Commons photographs

COMMONS_SOURCES = [  # (category, licence string, label)
    ("Category:Color photographs from the Farm Security Administration", LOC_FSA, "fsa"),
    ("Category:Sergey Prokudin-Gorsky", COMMONS_PD, "prokudin"),
    ("Category:Photographs by Sergey Prokudin-Gorsky", COMMONS_PD, "prokudin"),
    ("Category:United States Fish and Wildlife Service", FWS, "fws"),
]


def commons_files(http: Http, category: str, depth: int = 1) -> list[str]:
    """File titles of a category and (depth 1) of its direct sub-categories."""
    titles: list[str] = []
    for kind in ("file", "subcat"):
        token: dict[str, str] = {}
        while True:
            r = commons_api(
                http,
                action="query",
                list="categorymembers",
                cmtitle=category,
                cmtype=kind,
                cmlimit=500,
                **token,
            )
            members = [m["title"] for m in r["query"]["categorymembers"]]
            if kind == "file":
                titles += members
            elif depth > 0:
                for sub in members:
                    titles += commons_files(http, sub, depth - 1)
            if "continue" not in r:
                break
            token = {"cmcontinue": r["continue"]["cmcontinue"]}
    return titles


def commons_info(http: Http, titles: list[str]) -> dict[str, dict[str, Any]]:
    """imageinfo (url, size, mime, extmetadata) of file titles, cached by batch."""
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(titles), 40):
        batch = titles[i : i + 40]
        key = sha256_bytes("|".join(batch).encode())[:16]
        url = COMMONS + "?" + urllib.parse.urlencode(
            {
                "format": "json", "action": "query", "prop": "imageinfo", "titles": "|".join(batch),
                "iiprop": "url|size|mime|sha1|extmetadata",
                "iiextmetadatafilter": "LicenseShortName|Artist|ImageDescription|DateTimeOriginal|ObjectName|Categories",
            }
        )  # fmt: skip
        r = cached_json(http, url, f"commons_info_{key}")
        for page in r["query"]["pages"].values():
            if "imageinfo" in page:
                out[page["title"]] = page["imageinfo"][0]
    return out


def photo_stratum(title: str, info: dict[str, Any]) -> str | None:
    meta = {k: strip_html(v["value"]) for k, v in info["extmetadata"].items()}
    text = " ".join(
        [
            title,
            meta.get("ObjectName", ""),
            meta.get("ImageDescription", ""),
            meta.get("Categories", "").replace("|", " "),
        ]
    )
    if CHILD.search(text):
        return None
    if ANIMAL.search(text):
        return "photo_animal"
    if PEOPLE.search(text) and not OBJECT.search(title):
        return "photo_portrait"
    if OBJECT.search(text):
        return "photo_object"
    if SCENE.search(text):
        return "photo_scene"
    return None


def commons_candidates(http: Http) -> list[tuple[str, str, str, dict[str, Any]]]:
    """(title, stratum, licence, info) of the files that pass the licence, format, size and subject rules."""
    out: list[tuple[str, str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for category, licence, _label in COMMONS_SOURCES:
        titles = commons_files(http, category, depth=0)
        print(f"  {category}: {len(titles)} files", flush=True)
        info = commons_info(http, sorted(set(titles)))
        for title, ii in sorted(info.items()):
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            stem = re.sub(
                r"\s*(\(cropped\)|LCCN\d+|LOC \d+)\s*", "", title.rsplit(".", 1)[0]
            ).lower()
            if (
                title in seen or stem in seen
                or ii.get("mime") != "image/jpeg" or ii.get("size", 0) > MAX_FILE_BYTES
                or not eligible(ii["width"], ii["height"])
                or lic.lower() not in {"public domain", "cc0", "pd"} and not lic.lower().startswith("pd")
                or "cropped" in title.lower()
            ):  # fmt: skip
                continue
            stratum = photo_stratum(title, ii)
            if stratum is None:
                continue
            seen.update({title, stem})
            out.append((title, stratum, licence, ii))
    return out


def draw_photos(
    http: Http,
    rng: random.Random,
    tuning: list[np.ndarray],
    rejected: dict[str, str],
    have: dict[str, int],
) -> list[EvalImage]:
    cands = commons_candidates(http)
    print(
        f"  commons candidates: { {s: sum(1 for c in cands if c[1] == s) for s in QUOTAS['photo']} }",
        flush=True,
    )
    out: list[EvalImage] = []
    per_photographer: dict[str, int] = {}
    for stratum, quota in QUOTAS["photo"].items():
        pool = {c[0]: c for c in cands if c[1] == stratum}
        kept = have.get(stratum, 0)
        for title in seeded_order(list(pool), rng):
            if kept >= quota:
                break
            _, _, licence, ii = pool[title]
            fid = "commons_" + re.sub(
                r"[^a-z0-9]+", "_", title.rsplit(".", 1)[0].removeprefix("File:").lower()
            )[:60].strip("_")
            meta = {k: strip_html(v["value"]) for k, v in ii["extmetadata"].items()}
            author = meta.get("Artist", "") or "unknown"
            if fid in rejected or per_photographer.get(author, 0) >= MAX_PER_PHOTOGRAPHER:
                continue
            data = download_image(http, ii["url"], RAW / "source" / f"{fid}.jpg")
            with Image.open(io.BytesIO(data)) as im:
                ok, box = border_crop_box(im, lenient=True)
                if not ok:
                    continue
                pixels = prepare_image(im, box)
            if not is_colour(pixels) or near_tuning(pixels, tuning):
                continue
            tuning.append(thumb(pixels))
            per_photographer[author] = per_photographer.get(author, 0) + 1
            kept += 1
            out.append(
                EvalImage(
                    id=fid, category="photo", stratum=stratum, source="commons", dev=False, licence=licence, retrieved=TODAY,
                    sha256=sha256_bytes(data), prepared_sha256=pixel_sha256(pixels), title=meta.get("ObjectName", "") or title.removeprefix("File:"),
                    author=author, date=meta.get("DateTimeOriginal", "").split("<")[0].strip(),
                    page_url="https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                    url=ii["url"], source_id=title, width=ii["width"], height=ii["height"], colour=True, crop_box=box,
                    extra={"commons_sha1": ii["sha1"], "commons_licence": meta.get("LicenseShortName", "")},
                )
            )  # fmt: skip
            print(f"  {stratum}: {title[:70]!r} ({kept}/{quota})", flush=True)
    return out


# ----------------------------------------------------------------------------- SDXL entries


def sdxl_entries() -> list[EvalImage]:
    """The 30 SDXL entries (checksums are filled by ``scripts.generate_eval_sdxl --update``)."""
    spec = yaml.safe_load(SDXL_PROMPTS.read_text(encoding="utf-8"))
    out = []
    for p in spec["prompts"]:
        stratum = {"portrait": "sdxl_portrait", "scene": "sdxl_scene", "object": "sdxl_object"}[
            p["content"]
        ]
        out.append(
            EvalImage(
                id=f"sdxl_{p['index']:02d}", category="sdxl", stratum=stratum, source="sdxl", dev=False,
                licence="CC0 1.0 (own-generated, SDXL output)", retrieved="", sha256="", prepared_sha256="",
                title=p["prompt"], author="Stable Diffusion XL base 1.0 (own generation)", prompt=p["prompt"], seed=p["index"],
                subtype=p["style"], width=1024, height=1024,
            )
        )  # fmt: skip
    return out


# ----------------------------------------------------------------------------- stages


def cmd_curate(args: argparse.Namespace) -> None:
    from brushstroke.eval.dataset import draw_dev_split

    http = Http(args.insecure)
    rng = random.Random(DRAW_SEED)
    tuning = tuning_thumbs()
    rejected = load_rejections()
    print(f"{len(tuning)} tuning images, {len(rejected)} visual rejections", flush=True)
    images = draw_met(http, rng, tuning, rejected)
    have: dict[str, int] = {}
    for im in images:
        have[im.stratum] = have.get(im.stratum, 0) + 1
    images += draw_photos(http, rng, tuning, rejected, have)
    images += sdxl_entries()
    dev = draw_dev_split(images, rng)
    for im in images:
        im.dev = im.id in dev
    spec = yaml.safe_load(SDXL_PROMPTS.read_text(encoding="utf-8"))
    header = {
        "protocol": "docs/design/evaluation.md section 1 (T-022), curated by scripts/build_eval_set.py (T-024)",
        "draw_seed": DRAW_SEED,
        "curated": TODAY,
        "size": 512,
        "preparation": "centre crop to a square (or crop_box), Pillow BOX (area) filter to 512x512; prepared_sha256 = SHA-256 of the uint8 RGB pixels",
        "sdxl": spec["settings"],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    dump_manifest(MANIFEST, header, images)
    print(f"wrote {MANIFEST}: {len(images)} images, {len(dev)} dev", flush=True)


def cmd_sheets(args: argparse.Namespace) -> None:
    _, images = load_manifest(MANIFEST)
    out = Path("outputs/eval_set")
    out.mkdir(parents=True, exist_ok=True)
    cell, cols, rows = 256, 5, 4
    todo = [
        i for i in images if i.category != "sdxl" or (RAW / "prepared" / f"{i.id}.png").exists()
    ]
    for page, start in enumerate(range(0, len(todo), cols * rows)):
        sheet = Image.new("RGB", (cols * cell, rows * (cell + 14)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, im in enumerate(todo[start : start + cols * rows]):
            path = RAW / "prepared" / f"{im.id}.png"
            if not path.exists():
                src = next((RAW / "source").glob(f"{im.id}.*"), None)
                if src is None:
                    continue
                with Image.open(src) as s:
                    thumb_im = Image.fromarray(prepare_image(s, im.crop_box)).resize((cell, cell))
            else:
                thumb_im = Image.open(path).convert("RGB").resize((cell, cell))
            x, y = (k % cols) * cell, (k // cols) * (cell + 14)
            sheet.paste(thumb_im, (x, y + 14))
            draw.text((x + 2, y + 1), f"{im.id[-38:]}{' [dev]' if im.dev else ''}", fill="black")
        sheet.save(out / f"sheet_{page:02d}.png")
    print(f"sheets in {out}")


def cmd_check(args: argparse.Namespace) -> None:
    """No image of the set is, or looks like, an image used for tuning (T-009 sweep, M1/M2 demos)."""
    from brushstroke.eval.dataset import validate_manifest

    header, images = load_manifest(MANIFEST)
    problems = validate_manifest(header, images)
    tuning = tuning_thumbs()
    worst: dict[str, float] = {}
    for im in images:
        path = RAW / "prepared" / f"{im.id}.png"
        if not path.exists():
            raise SystemExit(f"{path} is missing: run python -m scripts.fetch_eval_set first")
        t = thumb(np.asarray(Image.open(path).convert("RGB")))
        worst[im.id] = max((float((t * u).mean()) for u in tuning), default=0.0)
    ids = {im.source_id for im in images if im.source == "met"}
    excluded = sorted(str(i) for i in MET_EXCLUDED if str(i) in ids)
    overlap = {k: round(v, 3) for k, v in worst.items() if v > 0.9}
    report = {
        "tuning_images": TUNING,
        "tuning_images_found": len(tuning),
        "images_checked": len(images),
        "max_thumbnail_correlation": round(max(worst.values()), 3),
        "near_duplicates_of_tuning_images": overlap,
        "excluded_met_objects_present": excluded,
        "manifest_problems": problems,
    }
    out = Path("docs/experiments/data/eval_set")
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlap_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    if overlap or excluded or problems:
        raise SystemExit(1)


def cmd_card(args: argparse.Namespace) -> None:
    """Write the composition tables of the data card (block ``EVALSET`` of ``docs/data_sources.md``)."""
    from collections import Counter

    from scripts.pt_repro_common import write_block

    _, images = load_manifest(MANIFEST)
    count = Counter((i.category, i.stratum, i.licence) for i in images)
    dev = Counter((i.category, i.stratum) for i in images if i.dev)
    lines = ["| category | stratum | n | dev | licence |", "|---|---|---|---|---|"]
    for (cat, stratum, lic), n in sorted(count.items()):
        lines.append(f"| {cat} | {stratum} | {n} | {dev.get((cat, stratum), 0)} | {lic} |")
    lines += [
        "",
        "| id | category / stratum | title | author | date | dev |",
        "|---|---|---|---|---|---|",
    ]
    for i in images:
        title = i.title.replace("|", "/")[:60]
        lines.append(
            f"| `{i.id}` | {i.category} / {i.stratum} | {title} | {i.author[:40].replace('|', '/')} | {i.date} | {'dev' if i.dev else ''} |"
        )
    write_block("EVALSET", chr(10).join(lines), Path("docs/data_sources.md"))
    print(f"{len(images)} rows")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("curate")
    c.add_argument("--insecure", action="store_true", help="skip TLS certificate verification")
    sub.add_parser("sheets")
    sub.add_parser("check")
    sub.add_parser("card")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]  # titles hold any script
    {"curate": cmd_curate, "sheets": cmd_sheets, "check": cmd_check, "card": cmd_card}[args.stage](
        args
    )


if __name__ == "__main__":
    main(sys.argv[1:])
