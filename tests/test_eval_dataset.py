"""Tests of the evaluation set (T-024): preparation, seeded draws, manifest validation, fetching."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brushstroke.eval import dataset as ds
from brushstroke.eval.dataset import EvalImage
from scripts import build_eval_set as build
from scripts import fetch_eval_set as fetch


def make_images() -> list[EvalImage]:
    """A synthetic manifest with the right counts per stratum (checksums are dummies)."""
    images = []
    for cat, strata in ds.QUOTAS.items():
        for stratum, n in strata.items():
            for k in range(n):
                idx = len(images)
                images.append(
                    EvalImage(
                        id=f"{stratum}_{k:02d}",
                        category=cat,
                        stratum=stratum,
                        source={"painting": "met", "photo": "commons", "sdxl": "sdxl"}[cat],
                        dev=False,
                        licence=(
                            ds.LICENCES[0]
                            if cat == "painting"
                            else ds.LICENCES[1]
                            if cat == "photo"
                            else ds.LICENCES[-1]
                        ),
                        retrieved="2026-10-26",
                        sha256=f"{idx:064x}",
                        prepared_sha256=f"{idx + 1000:064x}",
                        url="" if cat == "sdxl" else "https://example.org/x.jpg",
                        page_url="" if cat == "sdxl" else "https://example.org/x",
                        prompt="a prompt" if cat == "sdxl" else "",
                        seed=k + {"sdxl_portrait": 0, "sdxl_scene": 10, "sdxl_object": 20}[stratum]
                        if cat == "sdxl"
                        else None,
                    )
                )
    return images


def valid_set() -> tuple[dict[str, object], list[EvalImage]]:
    images = make_images()
    dev = ds.draw_dev_split(images, random.Random(ds.DRAW_SEED))
    for im in images:
        im.dev = im.id in dev
    return {"draw_seed": ds.DRAW_SEED}, images


# ------------------------------------------------------------------------ preparation


def test_the_box_filter_is_an_area_average_and_the_crop_is_centred() -> None:
    # 1536 px square made of constant 3 x 3 blocks: the area average of each block is its value
    src = np.zeros((1536, 1536, 3), dtype=np.uint8)
    blocks = np.arange(512 * 512, dtype=np.uint32).reshape(512, 512) % 251
    src[:] = np.kron(blocks, np.ones((3, 3), dtype=np.uint32)).astype(np.uint8)[..., None]
    out = ds.prepare_image(Image.fromarray(src))
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8
    assert np.array_equal(out[..., 0], blocks.astype(np.uint8))  # exact block average
    wide = np.zeros((1000, 1300, 3), dtype=np.uint8)
    wide[:, 150:1150] = 255  # the centred 1000 px square is white, the borders are black
    assert ds.prepare_image(Image.fromarray(wide)).min() == 255


def test_ineligible_sources_are_refused_unless_a_square_crop_box_is_given() -> None:
    small = Image.fromarray(np.zeros((700, 900, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="ineligible"):
        ds.prepare_image(small)
    assert not ds.eligible(900, 700) and not ds.eligible(2000, 800) and ds.eligible(1024, 768)
    assert ds.prepare_image(small, crop_box=(100, 0, 800, 700)).shape == (512, 512, 3)
    with pytest.raises(ValueError, match="square"):
        ds.prepare_image(small, crop_box=(0, 0, 800, 700))
    with pytest.raises(ValueError, match="outside"):
        ds.prepare_image(small, crop_box=(0, 0, 900, 900))


def test_greyscale_sources_become_three_equal_channels_and_colour_is_detected() -> None:
    grey = Image.fromarray(np.linspace(0, 255, 800 * 800, dtype=np.uint8).reshape(800, 800))
    out = ds.prepare_image(grey)
    assert np.array_equal(out[..., 0], out[..., 1]) and not ds.is_colour(out)
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[..., 0] = 200
    assert ds.is_colour(red)


def test_pixel_hash_depends_on_pixels_only() -> None:
    a = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert ds.pixel_sha256(a) == ds.pixel_sha256(a.copy()) != ds.pixel_sha256(a + 1)
    assert ds.pixel_sha256(np.asfortranarray(a)) == ds.pixel_sha256(a)  # layout-independent
    with pytest.raises(ValueError):
        ds.pixel_sha256(a.astype(np.float32))


# ------------------------------------------------------------------------ draws and manifest


def test_the_seeded_order_does_not_depend_on_the_input_order() -> None:
    ids = [5, 3, 9, 1, 7, 2]
    a = ds.seeded_order(ids, random.Random(ds.DRAW_SEED))
    b = ds.seeded_order(sorted(ids, reverse=True), random.Random(ds.DRAW_SEED))
    assert a == b and sorted(a) == sorted(ids)


def test_the_dev_split_has_the_planned_composition_and_is_deterministic() -> None:
    _, images = valid_set()
    dev = {im.id for im in images if im.dev}
    assert len(dev) == 10

    def count(cat: str, pred: Callable[[EvalImage], bool] = lambda im: True) -> int:
        return sum(1 for im in images if im.dev and im.category == cat and pred(im))

    assert count("painting") == 4 and count("photo") == 3 and count("sdxl") == 3
    assert count("painting", lambda im: im.stratum.startswith("european")) == 2
    assert count("painting", lambda im: im.stratum == "asian") == 1
    assert count("photo", lambda im: im.stratum == "photo_portrait") == 1
    assert sorted(im.seed for im in images if im.dev and im.category == "sdxl") == [4, 14, 25]
    again = ds.draw_dev_split(images, random.Random(ds.DRAW_SEED))
    assert again == dev


def test_a_valid_manifest_has_no_problems_and_each_defect_is_reported() -> None:
    header, images = valid_set()
    assert ds.validate_manifest(header, images) == []
    bad_count = images[1:]
    assert any("expected" in p for p in ds.validate_manifest(header, bad_count))
    images[0].licence = "All rights reserved"
    images[1].sha256 = ""
    images[2].id = images[3].id
    images[3].retrieved = ""
    problems = "\n".join(ds.validate_manifest(header, images))
    for needle in ("licence", "missing checksum", "duplicate ids", "retrieval date"):
        assert needle in problems
    _, grey = valid_set()
    for im in [i for i in grey if i.category == "photo"][:11]:
        im.colour = False
    assert any("greyscale" in p for p in ds.validate_manifest(header, grey))
    assert any("draw_seed" in p for p in ds.validate_manifest({"draw_seed": 1}, valid_set()[1]))


def test_manifest_round_trip(tmp_path: Path) -> None:
    header, images = valid_set()
    images[0].extra = {"medium": "Oil on canvas"}
    images[0].crop_box = [0, 0, 800, 800]
    path = tmp_path / "m.yaml"
    ds.dump_manifest(path, header, images)
    header2, images2 = ds.load_manifest(path)
    assert header2 == header and images2 == images


def test_the_committed_manifest_is_valid() -> None:
    path = Path("configs/eval/eval_set.yaml")
    if not path.exists():
        pytest.skip("manifest not curated yet")
    header, images = ds.load_manifest(path)
    assert ds.validate_manifest(header, images) == []
    assert len(images) == 100 and sum(im.dev for im in images) == 10


# ------------------------------------------------------------- curation rules and fetching


def test_the_met_strata_follow_the_subject_tags_and_the_classification() -> None:
    def obj(**kw: object) -> dict[str, object]:
        return {"isPublicDomain": True, "primaryImage": "u", "classification": "Paintings", **kw}

    tags = lambda *t: [{"term": x} for x in t]  # noqa: E731
    assert build.met_stratum(obj(tags=tags("Portraits", "Men")), "european") == (
        "european_portrait",
        "portrait",
    )
    assert build.met_stratum(obj(tags=tags("Landscapes")), "european") == (
        "european_landscape",
        "landscape",
    )
    assert build.met_stratum(obj(tags=tags("Still Life")), "european") == (
        "european_other",
        "other",
    )
    assert build.met_stratum(obj(tags=None), "european") is None  # no subject tag
    assert build.met_stratum(obj(isPublicDomain=False), "european") is None
    assert build.met_stratum(obj(primaryImage=""), "european") is None
    assert build.met_stratum(obj(classification="Prints", medium="Woodblock print"), "asian") == (
        "asian",
        "woodblock_print",
    )
    assert build.met_stratum(obj(classification="Prints", medium="Etching"), "asian") is None
    assert build.met_stratum(obj(), "modern") == ("other_art", "modern")


def test_the_photo_strata_exclude_children_and_use_the_subject_words() -> None:
    def info(**meta: str) -> dict[str, dict[str, dict[str, str]]]:
        return {"extmetadata": {k: {"value": v} for k, v in meta.items()}}

    assert build.photo_stratum("File:Mr. Smith, farmer.jpg", info()) == "photo_portrait"
    assert (
        build.photo_stratum("File:Boy with a dog.jpg", info()) is None
    )  # a child, whatever else is in the picture
    assert build.photo_stratum("File:Mules at the fair.jpg", info()) == "photo_animal"
    assert build.photo_stratum("File:Truck on a dirt road.jpg", info()) == "photo_object"
    assert build.photo_stratum("File:Farm in the valley.jpg", info()) == "photo_scene"
    assert build.photo_stratum("File:Untitled.jpg", info(Categories="Misc")) is None


def test_near_duplicates_of_tuning_images_are_detected() -> None:
    rng = np.random.default_rng(0)
    tuning = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    thumbs = [build.thumb(tuning)]
    assert build.near_tuning(tuning, thumbs)
    assert build.near_tuning(np.clip(tuning.astype(np.int16) + 10, 0, 255).astype(np.uint8), thumbs)
    assert not build.near_tuning(other, thumbs)


def test_fetch_verifies_checksums_and_prepares_the_image(tmp_path: Path) -> None:
    arr = (np.random.default_rng(1).random((900, 1000, 3)) * 255).astype(np.uint8)
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    path = src_dir / "t1.jpg"
    Image.fromarray(arr).save(path, quality=95)
    pixels = ds.prepare_image(Image.open(path))
    im = EvalImage(
        id="t1",
        category="photo",
        stratum="photo_scene",
        source="commons",
        dev=False,
        licence=ds.LICENCES[1],
        retrieved="2026-10-26",
        sha256=ds.sha256_file(path),
        prepared_sha256=ds.pixel_sha256(pixels),
        url="https://example.org/t1.jpg",
        page_url="https://example.org/t1",
    )
    assert fetch.fetch_one(im, tmp_path, None) == "ok"
    assert np.array_equal(np.asarray(Image.open(tmp_path / "prepared" / "t1.png")), pixels)
    im.sha256 = (
        "0" * 64
    )  # the bytes changed but the decoded pixels did not (a re-encoded museum file)
    assert fetch.fetch_one(im, tmp_path, None).startswith("ok (file bytes differ")
    im.sha256 = ds.sha256_file(path)
    im.prepared_sha256 = "0" * 64
    assert "prepared" in fetch.fetch_one(im, tmp_path, None)
    path.unlink()
    assert fetch.fetch_one(im, tmp_path, None) == "missing"


def test_photographer_names_are_normalized_before_the_cap_is_counted() -> None:
    same = [
        "Russell Lee",
        "Lee, Russell, 1903-1986, photographer",
        "Russell Lee (American, 1903-1986)",
    ]
    assert {frozenset(build.photographers(a)) for a in same} == {frozenset({"lee russell"})}
    both = build.photographers(
        "Delano, Jack, photographer; Wolcott, Marion Post, 1910-1990, photographer"
    )
    assert both == {"delano jack", "marion post wolcott"}
    assert build.photographers("Marion Post Wolcott") <= both
    assert build.photographers("") == {"unknown"}
    assert build.photographers("U.S. Fish and Wildlife Service") == {"fish service wildlife"}


def test_two_views_of_one_scene_are_one_series() -> None:
    def view(title: str, cats: set[str], who: str = "Louise Rosskam", agency: bool = False) -> dict:
        return {
            "photographers": build.photographers(who),
            "categories": cats,
            "title": title,
            "agency": agency,
        }

    shared = {
        "Southwest, Washington, D.C.",
        "Demolished buildings in Washington, D.C.",
        "PD US FSA/OWI",
    }
    a = view("File:Rosskam-dc1.jpg", shared)
    b = view("File:Shulman's Market at the corner.jpg", shared | {"1931 Chevrolet automobiles"})
    assert build.same_series(a, b, frequent=set())
    assert not build.same_series(
        a, view("File:Other.jpg", shared, who="Jack Delano"), frequent=set()
    )
    generic = {"PD US FSA/OWI", "Color photographs from the Farm Security Administration"}
    assert not build.same_series(
        view("File:A one.jpg", generic), view("File:Zzz.jpg", generic), frequent=set()
    )
    assert not build.same_series(
        a, b, frequent=set(shared)
    )  # categories shared by many files are not specific
    assert build.same_series(
        view("File:Barn, Vermont LCCN1.jpg", set()),
        view("File:Barn, Vermont LCCN2.jpg", set()),
        frequent=set(),
    )
    assert not build.same_series(
        a, view("File:Rosskam-dc2.jpg", shared, agency=True), frequent=set()
    )  # agency photos


def test_commons_dates_lose_the_wikidata_residue() -> None:
    raw = 'August 1940<div style="display: none;">date QS:P571,+1940-08-00T00:00:00Z/10</div>'
    assert build.clean_date(raw) == "August 1940"
    assert build.clean_date("1941-02") == "1941-02"
    assert build.clean_date("Taken on 22 June 2007, 14:32:20") == "22 June 2007, 14:32:20"


def test_a_redraw_keeps_the_sdxl_generation_results(tmp_path: Path) -> None:
    header, images = valid_set()
    old = tmp_path / "old.yaml"
    ds.dump_manifest(old, header, images)
    fresh = [
        EvalImage(
            id=im.id, category="sdxl", stratum=im.stratum, source="sdxl", dev=False,
            licence=im.licence, retrieved="", sha256="", prepared_sha256="",
            prompt="p", seed=im.seed,
        )
        for im in images
        if im.category == "sdxl"
    ]  # fmt: skip
    kept = build.carry_over_sdxl(fresh, old)
    originals = {im.id: im for im in images}
    assert all(k.sha256 == originals[k.id].sha256 and k.retrieved == "2026-10-26" for k in kept)
    assert build.carry_over_sdxl(fresh, tmp_path / "missing.yaml") == fresh


def test_a_mismatch_of_a_downloaded_image_is_not_masked_by_the_sdxl_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    header, images = valid_set()
    monkeypatch.setattr(fetch, "load_manifest", lambda path: (header, images))
    monkeypatch.setattr(fetch, "validate_manifest", lambda h, i: [])
    monkeypatch.setattr(
        fetch,
        "fetch_one",
        lambda im, out, http: (
            "SHA-256 MISMATCH (prepared pixels)"
            if im.id.endswith("_00") and im.category == "painting"
            else "ok"
        ),
    )
    monkeypatch.setattr(fetch, "verify_sdxl", lambda im, out: "ok")
    monkeypatch.setattr(fetch.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(SystemExit) as exc:
        fetch.main(["--out", str(tmp_path), "--sdxl"])
    assert exc.value.code == 1  # the SDXL regeneration succeeded, the paintings still fail
    monkeypatch.setattr(fetch, "fetch_one", lambda im, out, http: "ok")
    fetch.main(["--out", str(tmp_path), "--sdxl"])  # all ok: no exit


def test_a_detected_person_is_what_has_person_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    def detector(scores: list[float], labels: list[int]) -> object:
        return lambda batch: [{"scores": torch.tensor(scores), "labels": torch.tensor(labels)}]

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(build, "person_detector", lambda: detector([0.9, 0.2], [1, 1]))
    assert build.has_person(img)
    monkeypatch.setattr(build, "person_detector", lambda: detector([0.3], [1]))
    assert not build.has_person(img)  # below the threshold
    monkeypatch.setattr(build, "person_detector", lambda: detector([0.95], [18]))
    assert not build.has_person(img)  # a dog is not a person
    monkeypatch.setattr(build, "person_detector", lambda: detector([], []))
    assert not build.has_person(img)


def test_a_thin_light_edge_line_in_front_of_a_dark_film_band_is_seen_through() -> None:
    img = np.full((512, 512, 3), 160, dtype=np.uint8)
    img[:, :18] = 8  # an 18 px black film edge on the left ...
    img[:, 0] = 200  # ... behind a 1 px light line at the very edge
    img[:21] = 8
    img[0] = 180
    assert ds.border_area_fraction(img, "film") == pytest.approx(
        (21 * 512 + 18 * (512 - 21)) / 512**2
    )
    big = Image.fromarray(np.kron(img, np.ones((4, 4, 1), dtype=np.uint8)))  # 2048 px source
    ok, box = ds.border_crop_box(big, kind="film")
    assert ok and box is not None and box[0] >= 18 * 4 and box[1] >= 21 * 4
    plain = Image.fromarray(np.full((900, 900, 3), 150, dtype=np.uint8))
    assert ds.border_crop_box(plain, kind="film") == (True, None)
    assert ds.edge_band(np.array([False, False, False, False, True, True]), 6) == 0
    assert ds.edge_band(np.array([False, True, True, False]), 6) == 3


def test_the_film_rule_sees_a_soft_inner_edge_and_the_flat_rule_never_trims_texture() -> None:
    img = np.full((512, 512, 3), 140, dtype=np.uint8)
    img[:, :12] = 4
    img[:, 12:20] = 25  # the soft inner edge of a film bar: mean luminance below 30
    img[:, 20:24] = 90
    assert ds.border_area_fraction(img, "film") == pytest.approx(20 / 512, abs=1e-3)
    textured = np.random.default_rng(0).integers(0, 40, (512, 512, 3), dtype=np.uint8)
    assert ds.border_area_fraction(textured, "flat") == 0.0  # dark but textured: content
    with pytest.raises(ValueError):
        ds.border_flags(np.zeros((4, 4), dtype=np.float32), "nope")


def test_the_agency_word_filter_is_a_valid_pattern_that_matches_the_words() -> None:
    assert not any(ord(c) < 32 for c in build.FWS_PEOPLE.pattern)  # no control character
    for text in ("Staff at work", "Oiled bird cleaning", "gannets in rehab", "rehabilitation"):
        assert build.FWS_PEOPLE.search(text), text
    for text in ("manatee", "Mandarin duck", "Dayton marsh", "Haliaeetus leucocephalus"):
        assert not build.FWS_PEOPLE.search(text), text


def test_no_text_file_of_the_repository_holds_a_control_character() -> None:
    suffixes = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".cfg"}
    skip = {".venv", "outputs", "data", ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
    bad = []
    for path in Path(".").rglob("*"):
        if not path.is_file() or path.suffix not in suffixes or set(path.parts) & skip:
            continue
        if any(b < 32 and b not in (9, 10, 12, 13) for b in path.read_bytes()):
            bad.append(str(path))
    assert bad == []


def audit_images() -> list[EvalImage]:
    _, images = valid_set()
    for im in images:
        if im.source == "commons":
            im.author = "Russell Lee"
            im.source_id = f"File:{im.id}.jpg"
            im.url = "https://upload.wikimedia.org/wikipedia/commons/x.jpg"
        elif im.source == "met":
            im.url = "https://images.metmuseum.org/CRDImages/x.jpg"
    return images


def test_the_audit_flags_a_photographer_over_the_cap_and_a_wrong_dev_split() -> None:
    from scripts import audit_eval_set as audit

    images = audit_images()
    counts, bad = audit.check_photographers(images)
    assert counts["lee russell"] == sum(1 for im in images if im.source == "commons")
    assert len(bad) == 1 and "lee russell" in bad[0]
    assert audit.check_dev_test(images) == []
    images[0].dev = not images[0].dev
    assert any("dev" in b for b in audit.check_dev_test(images))


def test_the_audit_flags_unclean_sources_a_foreign_host_and_a_child_title() -> None:
    from scripts import audit_eval_set as audit

    images = audit_images()
    assert audit.check_sources(images) == []
    commons = next(im for im in images if im.source == "commons")
    commons.date = "May 1942date QS:P571,+1942-05-00T00:00:00Z/10"
    commons.url = "https://example.org/x.jpg"
    commons.title = "Boy with a dog\x08"
    commons.licence = ds.LICENCES[1]
    problems = "\n".join(audit.check_sources(images))
    assert "unclean date" in problems and "unclean title" in problems and "allow-list" in problems


def test_the_audit_sees_two_copies_of_one_image(tmp_path: Path) -> None:
    from scripts import audit_eval_set as audit

    images = audit_images()[:3]
    rng = np.random.default_rng(3)
    (tmp_path / "prepared").mkdir()
    base = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    arrays = [base, base.copy(), rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)]
    for im, arr in zip(images, arrays, strict=True):
        Image.fromarray(arr).save(tmp_path / "prepared" / f"{im.id}.png")
    worst, bad = audit.check_duplicates(tmp_path, images)
    assert worst > 0.99 and any("correlate" in b for b in bad)


def test_colliding_commons_ids_get_a_hash_and_cannot_share_a_cache_file() -> None:
    long = "File:First snow of the season in the foothills of the Little Belt Mountains"
    titles = [long + " LCCN1.jpg", long + "1a35029v.jpg", "File:A short title.jpg"]
    fids = build.unique_fids(titles)
    assert len(set(fids.values())) == 3 and fids[titles[2]] == "commons_a_short_title"
    assert all(len(v) <= 68 for v in fids.values()) and fids[titles[0]] != fids[titles[1]]
    assert fids == build.unique_fids(titles[::-1])  # a function of the title set, not of the order


def test_a_cache_file_that_is_not_the_described_file_is_downloaded_again(tmp_path: Path) -> None:
    class FakeHttp:
        calls = 0

        def get(self, url: str) -> bytes:
            self.calls += 1
            return b"the right file"

    dest = tmp_path / "x.jpg"
    dest.write_bytes(b"another file with the same cache name")
    http = FakeHttp()
    right = build.sha1_hex(b"the right file")
    assert build.download_image(http, "https://x/y.jpg", dest, right) == b"the right file"  # type: ignore[arg-type]
    assert http.calls == 1 and dest.read_bytes() == b"the right file"
    assert build.download_image(http, "https://x/y.jpg", dest, right) == b"the right file"  # type: ignore[arg-type]
    assert http.calls == 1  # now cached and verified


def test_fetch_reports_a_changed_source_instead_of_crashing(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir()
    Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8)).save(tmp_path / "source" / "t2.jpg")
    im = EvalImage(
        id="t2", category="photo", stratum="photo_scene", source="commons", dev=False,
        licence=ds.LICENCES[1], retrieved="2026-10-26", sha256="0" * 64, prepared_sha256="1" * 64,
        url="https://example.org/t2.jpg", page_url="https://example.org/t2",
        crop_box=[10, 10, 900, 900],
    )  # fmt: skip
    status = fetch.fetch_one(im, tmp_path, None)
    assert status.startswith("SHA-256 MISMATCH") and "outside" in status
