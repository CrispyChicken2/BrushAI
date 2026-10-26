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
import difflib
import functools
import hashlib
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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from brushstroke.eval.dataset import (
    DRAW_SEED,
    QUOTAS,
    EvalImage,
    border_crop_box,
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
COLOUR_OVERRIDES = {
    "met_263532": True
}  # hand-coloured albumen print (contact sheet), not monochrome
MET_EXCLUDED = {45434}  # The Great Wave off Kanagawa (Met impression), in the T-009 sweep
TODAY = time.strftime("%Y-%m-%d")
MET_LICENCE = "CC0 1.0 (Met Open Access, isPublicDomain)"
LOC_FSA = "Public domain (US government work, Library of Congress FSA/OWI)"
FWS = "Public domain (US government work, US Fish and Wildlife Service)"
COMMONS_PD = "Public domain (Wikimedia Commons licence tag)"
MAX_PER_PHOTOGRAPHER = 4
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
    r"bottle|store|church|pump|tool|windmill|silo|tent|trailer|bicycle|gate|fence|shed|shack|cabin|poster|"
    r"jars?|barrel|basket|chair|door|window|stove|clock|radio|lamp|table|furniture|quilt|doll|wheel|bucket|"
    r"crate|sack|hat|shoes?|saddle|harness|plane|airplane|biplane|market|shop|barbershop|laundry)\b",
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


# of the 10 photographic portraits: monochrome Met prints (at most 10 greyscale photographs in the set)
MET_PHOTO_PORTRAITS = 9
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


def sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()  # noqa: S324  (the checksum Commons publishes, not a security use)


def download_image(http: Http, url: str, dest: Path, sha1: str | None = None) -> bytes:
    """The bytes of ``url``, cached in ``dest``. With ``sha1`` (Commons publishes one per file) a cached file that is
    not that file is downloaded again: a cache name must never stand for another file."""
    if dest.exists():
        data = dest.read_bytes()
        if sha1 is None or sha1_hex(data) == sha1:
            return data
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
                ok, box = border_crop_box(im, kind="flat")
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
                    colour=COLOUR_OVERRIDES.get(f"met_{oid}", not found[0].startswith("photo")),
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
    # Prokudin-Gorsky files were tried and dropped: most are sepia-toned reproductions, not colour photographs
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


@functools.lru_cache(maxsize=1)
def person_detector() -> Any:
    """COCO Faster R-CNN (MobileNetV3, torchvision) used only to keep people out of the modern agency photographs."""
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )

    return fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    ).eval()


def has_person(pixels: np.ndarray, threshold: float = 0.4) -> bool:
    """True if a COCO "person" is detected with a score above ``threshold``, whatever its size or position."""
    import torch

    img = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        out = person_detector()([img])[0]
    return bool(((out["labels"] == 1) & (out["scores"] > threshold)).any())


def photo_text(title: str, info: dict[str, Any]) -> str:
    meta = {k: strip_html(v["value"]) for k, v in info["extmetadata"].items()}
    return " ".join(
        [
            title,
            meta.get("ObjectName", ""),
            meta.get("ImageDescription", ""),
            meta.get("Categories", "").replace("|", " "),
        ]
    ).replace("_", " ")


# a species category of Commons ("Haliaeetus leucocephalus"): the subject is an animal, not an event
FWS_BINOMIAL = re.compile(r"(^|\|)[A-Z][a-z]+ [a-z]+( [a-z]+)?(\||$)")
FWS_PEOPLE_WORDS = [
    "staff",
    "volunteers?",
    "visitors?",
    "biologists?",
    "crew",
    "team",
    "ceremony",
    "release",
    "released",
    "releasing",
    "event",
    "events",
    "day",
    "days",
    "workers?",
    "photographer",
    "group",
    "wedding",
    "officers?",
    "agents?",
    "director",
    "chief",
    "students?",
    "people",
    "person",
    "man",
    "woman",
    "men",
    "women",
    "kids?",
    "child",
    "children",
    "boy",
    "girl",
    "family",
    "tour",
    "festival",
    "celebration",
    "employees?",
    "interns?",
    "researchers?",
    "hunters?",
    "anglers?",
    "fishermen",
    "fisherman",
    "public",
    "crowd",
    "portrait",
    "handler",
    "holding",
    "handling",
    "banding",
    "survey",
    "sampling",
    "monitoring",
    "collecting",
    "tagging",
    "stocking",
    "food",
    "fried",
    "cooking",
    "recipe",
    "searches",
    "searching",
    "oiled",
    "rescue",
    "rehabilitation",
    "scientists?",
    "rangers?",
    "demonstration",
    "banded",
    "manager",
    "office",
    "meeting",
    "award",
    "interview",
    "conference",
    "training",
    "program",
    "programme",
    "hands?",
    "capture",
    "captured",
    "netting",
    "trapping",
]
FWS_PEOPLE_WORDS += ["rehab", "briefings?", "cleaning", "spill", "response"]
# titles about something other than a visible animal: its nest, a fire, a habitat, traces
FWS_PEOPLE_WORDS += [
    "nests?",
    "burn",
    "burns",
    "fire",
    "fires",
    "smoke",
    "habitat",
    "tracks?",
    "scat",
    "eggs?",
    "skull",
    "pelt",
    "footprints?",
    "survey",
]
FWS_PEOPLE = re.compile(r"\b(" + "|".join(FWS_PEOPLE_WORDS) + r")\b", re.I)


def photo_stratum(title: str, info: dict[str, Any]) -> str | None:
    text = photo_text(title, info)
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


GENERIC_CATEGORY = re.compile(
    r"Farm Security Administration|Library of Congress|PD US|photographs? (by|of|from)|Images uploaded|"
    r"\b(19|20)\d\d\b|United States|Color photographs|public domain|Files with|Files from",
    re.I,
)
MAX_SERIES_CATEGORY = (
    30  # a category shared by more candidates than this is not specific to one place or event
)
SERIES_TITLE_RATIO = 0.6


def photographers(author: str) -> set[str]:
    """Normalized photographer names of a Commons ``Artist`` string, so that "Lee, Russell, 1903-1986, photographer"
    and "Russell Lee" are the same person: lower case, no dates or "photographer", word order ignored."""
    names: set[str] = set()
    for part in re.split(r"[;/]", author):
        part = re.sub(
            r"\([^)]*\)|\b(photographer|photo|by)\b|\d{4}(-\d{4})?", " ", part, flags=re.I
        )
        tokens = [
            w
            for w in re.split(r"[\s,.]+", part.lower())
            if len(w) > 1 and w not in {"and", "the", "of"}
        ]
        if tokens:
            names.add(" ".join(sorted(tokens)))
    return names or {"unknown"}


def title_stem(title: str) -> str:
    return re.sub(
        r"\s*(LCCN\d+|LOC \d+|\(cropped\))", "", title.removeprefix("File:").rsplit(".", 1)[0]
    ).lower()


def same_series(a: dict[str, Any], b: dict[str, Any], frequent: set[str]) -> bool:
    """Two Commons photographs of the same photographer that share a place or event (a category that is neither generic
    nor shared by many candidates) or have similar titles are views of one scene: only one may be in the set."""
    if a.get("agency") or b.get("agency"):
        return False  # agency wildlife photographs share species categories, not scenes
    if not a["photographers"] & b["photographers"]:
        return False
    shared = {
        c
        for c in a["categories"] & b["categories"]
        if c not in frequent and not GENERIC_CATEGORY.search(c)
    }
    ratio = difflib.SequenceMatcher(None, title_stem(a["title"]), title_stem(b["title"])).ratio()
    return bool(shared) or ratio >= SERIES_TITLE_RATIO


def clean_date(raw: str) -> str:
    """The date of a Commons file without the Wikidata residue ("date QS:P571,...") or the "Taken on" prefix."""
    text = re.split(r"date QS", strip_html(raw))[0].replace(" ", " ").strip()
    return re.sub(r"^Taken on\s+", "", text)


def unique_fids(titles: list[str]) -> dict[str, str]:
    """Manifest id of each Commons title: ``commons_`` and the first 60 characters of the title, made of letters and
    digits. Titles whose ids would collide (long titles of one series differ only at the end) get a hash suffix, so
    that an id, and the cache file named after it, always stand for one file."""
    base = {
        t: "commons_"
        + re.sub(r"[^a-z0-9]+", "_", t.rsplit(".", 1)[0].removeprefix("File:").lower())[:60].strip(
            "_"
        )
        for t in titles
    }
    count = Counter(base.values())
    return {
        t: b
        if count[b] == 1
        else f"{b[:52].rstrip('_')}_{hashlib.sha1(t.encode()).hexdigest()[:7]}"
        for t, b in base.items()
    }


def commons_candidates(http: Http) -> list[tuple[str, str, str, dict[str, Any]]]:
    """(title, stratum, licence, info) of the files that pass the licence, format, size and subject rules."""
    out: list[tuple[str, str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for category, licence, label in COMMONS_SOURCES:
        titles = commons_files(http, category, depth=1 if label == "fws" else 0)
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
            if label == "fws" and (
                stratum != "photo_animal"
                or not ANIMAL.search(title.replace("_", " "))  # the animal is named in the title
                or not FWS_BINOMIAL.search(ii["extmetadata"].get("Categories", {}).get("value", ""))
                or FWS_PEOPLE.search(photo_text(title, ii))
            ):
                continue  # modern agency photographs: only wildlife, never anything that mentions people or events
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
    fids = unique_fids([c[0] for c in cands])
    per_photographer: dict[str, int] = {}
    cat_count: dict[str, int] = {}
    for c in cands:
        for name in set(
            strip_html(c[3]["extmetadata"].get("Categories", {}).get("value", "")).split("|")
        ):
            cat_count[name] = cat_count.get(name, 0) + 1
    frequent = {c for c, n in cat_count.items() if n > MAX_SERIES_CATEGORY}
    accepted: list[dict[str, Any]] = []
    for stratum, quota in QUOTAS["photo"].items():
        pool = {c[0]: c for c in cands if c[1] == stratum}
        kept = have.get(stratum, 0)
        for title in seeded_order(list(pool), rng):
            if kept >= quota:
                break
            _, _, licence, ii = pool[title]
            fid = fids[title]
            meta = {k: strip_html(v["value"]) for k, v in ii["extmetadata"].items()}
            author = meta.get("Artist", "") or "unknown"
            who = photographers(author)
            view = {
                "photographers": who,
                "categories": set(meta.get("Categories", "").split("|")),
                "title": title,
                "agency": licence == FWS,
            }
            if (
                fid in rejected
                or any(per_photographer.get(n, 0) >= MAX_PER_PHOTOGRAPHER for n in who)
                or any(same_series(view, other, frequent) for other in accepted)
            ):
                continue
            data = download_image(http, ii["url"], RAW / "source" / f"{fid}.jpg", ii["sha1"])
            if sha1_hex(data) != ii["sha1"]:
                continue  # the file served is not the one the API describes (re-uploaded since): unusable
            with Image.open(io.BytesIO(data)) as im:
                src_w, src_h = (
                    im.size
                )  # the size of the file used, not the (possibly stale) API field
                ok, box = border_crop_box(im, kind="film" if licence == LOC_FSA else "flat")
                if not ok:
                    continue
                pixels = prepare_image(im, box)
            if not is_colour(pixels) or near_tuning(pixels, tuning):
                continue
            if licence == FWS and has_person(pixels):
                continue  # modern photographs: nobody who may be alive, detected, not only judged by eye
            tuning.append(thumb(pixels))
            for n in who:
                per_photographer[n] = per_photographer.get(n, 0) + 1
            accepted.append(view)
            kept += 1
            out.append(
                EvalImage(
                    id=fid, category="photo", stratum=stratum, source="commons", dev=False, licence=licence, retrieved=TODAY,
                    sha256=sha256_bytes(data), prepared_sha256=pixel_sha256(pixels), title=meta.get("ObjectName", "") or title.removeprefix("File:"),
                    author=author, date=clean_date(ii["extmetadata"].get("DateTimeOriginal", {}).get("value", "")),
                    page_url="https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                    url=ii["url"], source_id=title, width=src_w, height=src_h, colour=True, crop_box=box,
                    extra={"commons_sha1": ii["sha1"], "commons_licence": meta.get("LicenseShortName", ""), "commons_categories": meta.get("Categories", "")},
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


def _old_header() -> dict[str, Any]:
    return load_manifest(MANIFEST)[0] if MANIFEST.exists() else {}


def carry_over_sdxl(entries: list[EvalImage], previous: Path) -> list[EvalImage]:
    """Keep the generation results (hashes, retrieval date, regeneration log) of an existing manifest: the SDXL images
    do not depend on the draw, and a redraw of the paintings and photographs must not erase them."""
    if not previous.exists():
        return entries
    _, old = load_manifest(previous)
    by_id = {im.id: im for im in old if im.category == "sdxl"}
    for im in entries:
        if im.id in by_id:
            keep = by_id[im.id]
            im.sha256, im.prepared_sha256, im.retrieved, im.extra = (
                keep.sha256,
                keep.prepared_sha256,
                keep.retrieved,
                keep.extra,
            )
    return entries


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
    images += carry_over_sdxl(sdxl_entries(), MANIFEST)
    dev = draw_dev_split(images, rng)
    for im in images:
        im.dev = im.id in dev
    spec = yaml.safe_load(SDXL_PROMPTS.read_text(encoding="utf-8"))
    old_versions = _old_header().get("sdxl", {}).get("versions")
    sdxl_header = {**spec["settings"], **({"versions": old_versions} if old_versions else {})}
    header = {
        "protocol": "docs/design/evaluation.md section 1 (T-022), curated by scripts/build_eval_set.py (T-024)",
        "draw_seed": DRAW_SEED,
        "curated": TODAY,
        "size": 512,
        "preparation": "centre crop to a square (or crop_box), Pillow BOX (area) filter to 512x512; prepared_sha256 = SHA-256 of the uint8 RGB pixels",
        "sdxl": sdxl_header,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    dump_manifest(MANIFEST, header, images)
    print(f"wrote {MANIFEST}: {len(images)} images, {len(dev)} dev", flush=True)


def cmd_sheets(args: argparse.Namespace) -> None:
    _, images = load_manifest(MANIFEST)
    out = Path("outputs/eval_set")
    out.mkdir(parents=True, exist_ok=True)
    cell, cols, rows = args.cell, args.cols, args.rows
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
            label = f"{start + k + 1:03d} {im.id[-36:]}{' [dev]' if im.dev else ''}"
            draw.text((x + 2, y + 1), label, fill="black")
        sheet.save(out / f"sheet_{page:02d}.png")
    print(f"sheets in {out}")


def frequent_categories(items: list[dict[str, Any]]) -> set[str]:
    count: dict[str, int] = {}
    for it in items:
        for c in it["categories"]:
            count[c] = count.get(c, 0) + 1
    return {
        c for c, n in count.items() if n > 3
    }  # in the final set a category shared by more than 3 images is generic


def photographer_counts(images: list[EvalImage]) -> dict[str, int]:
    count: dict[str, int] = {}
    for im in images:
        if im.source == "commons":
            for n in photographers(im.author):
                count[n] = count.get(n, 0) + 1
    return dict(sorted(count.items(), key=lambda kv: -kv[1]))


def cmd_check(args: argparse.Namespace) -> None:
    """No image of the set is, or looks like, an image used for tuning (T-009 sweep, M1/M2 demos). Every other
    machine-checkable rule is in ``scripts/audit_eval_set.py``."""
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
    shown: set[tuple[str, str]] = set()
    for (cat, stratum, lic), n in sorted(count.items()):
        n_dev = "" if (cat, stratum) in shown else str(dev.get((cat, stratum), 0))
        shown.add((cat, stratum))
        lines.append(f"| {cat} | {stratum} | {n} | {n_dev} | {lic} |")
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
    sh = sub.add_parser("sheets")
    sh.add_argument("--cell", type=int, default=320)
    sh.add_argument("--cols", type=int, default=4)
    sh.add_argument("--rows", type=int, default=3)
    sub.add_parser("check")
    sub.add_parser("card")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]  # titles hold any script
    {"curate": cmd_curate, "sheets": cmd_sheets, "check": cmd_check, "card": cmd_card}[args.stage](
        args
    )


if __name__ == "__main__":
    main(sys.argv[1:])
