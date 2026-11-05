"""Build the Learning to Paint training subset from Met Open Access (T-029).

    python -m scripts.build_l2p_trainset plan     # pin the CSV, exclusions, candidate counts per stratum (no image download)
    python -m scripts.build_l2p_trainset fetch    # Met API + image download -> data/raw/l2p_met/chunks (resumable)
    python -m scripts.build_l2p_trainset verify   # trim, near-duplicate and disjointness checks, images.npy, ids and summary files
    python -m scripts.build_l2p_trainset check    # re-run the disjointness checks on the stored subset (fast, no network)

Settings: ``configs/l2p/trainset.yaml`` (``brushstroke.baselines.l2p.trainset.TrainsetConfig``). Images and chunks are written
to ``data/raw/l2p_met`` (outside git); the committed outputs are the id list, the exclusion record and the summary.
See ``docs/design/learning_to_paint.md`` section 7 and ``docs/data_sources.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from brushstroke.baselines.l2p import trainset as ts
from brushstroke.utils.config import load_config
from brushstroke.utils.http import Http

CONFIG = Path("configs/l2p/trainset.yaml")
CHUNK = 1000  # candidates per resumable chunk
SUMMARY = Path("docs/experiments/data/l2p/trainset_summary.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def candidates(cfg: ts.TrainsetConfig) -> tuple[list[dict[str, Any]], set[int], set[str]]:
    root = Path(cfg.root)
    csv_path = root / cfg.csv_name
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} missing: download {cfg.csv_url}")
    if cfg.csv_sha256 and sha256_file(csv_path) != cfg.csv_sha256:
        raise SystemExit(f"{csv_path}: SHA-256 differs from the pinned one")
    ids, artists = ts.exclusions(Path(cfg.eval_manifest))
    return ts.plan(list(ts.read_candidates(csv_path)), cfg, ids, artists), ids, artists


def cmd_plan(cfg: ts.TrainsetConfig) -> None:
    order, ids, artists = candidates(cfg)
    print(
        json.dumps(
            {
                "csv_sha256": sha256_file(Path(cfg.root) / cfg.csv_name),
                "eval_met_objects": len(ids),
                "eval_artists": len(artists),
                "candidates": len(order),
                "per_stratum": ts.class_counts(order),
            },
            indent=1,
        )
    )


def fetch_one(http: Http, cand: dict[str, Any], artists: set[str]) -> dict[str, Any]:
    """Download and prepare the image of one candidate: a dict with ``ok`` and either the pixels or the rejection reason."""
    out: dict[str, Any] = {"id": cand["id"], "stratum": cand["stratum"], "ok": False}
    if set(cand["artists"]) & artists:
        out["reason"] = "eval_artist"
        return out
    try:
        data = http.get(cand["url"], retries=3)
    except Exception as err:  # noqa: BLE001
        out["reason"] = f"image:{type(err).__name__}"
        return out
    arr, reason = ts.crop_resize(data)
    if arr is None:
        out["reason"] = reason
        return out
    out.update(
        ok=True,
        pixels=arr,
        pixel_sha256=ts.pixel_sha256(arr),
        file_sha256=hashlib.sha256(data).hexdigest(),
        **{k: cand[k] for k in ("url", "title", "artist", "date", "classification", "department")},
    )
    return out


def cmd_fetch(cfg: ts.TrainsetConfig) -> None:
    order, _, artists = candidates(cfg)
    root = Path(cfg.root)
    chunks = root / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    http = Http(pause=0.0, timeout=20.0)
    accepted = sum(int(json.loads(p.read_text())["accepted"]) for p in chunks.glob("chunk_*.json"))
    t0 = time.time()
    for k in range(0, len(order), CHUNK):
        if accepted >= cfg.target:
            break
        name = f"chunk_{k // CHUNK:05d}"
        if (chunks / f"{name}.json").exists():
            continue
        with ThreadPoolExecutor(cfg.threads) as pool:
            results = list(pool.map(lambda c: fetch_one(http, c, artists), order[k : k + CHUNK]))
        good = [r for r in results if r["ok"]]
        np.savez(
            chunks / f"{name}.npz",
            pixels=np.stack([r["pixels"] for r in good])
            if good
            else np.zeros((0, ts.SIZE, ts.SIZE, 3), np.uint8),
        )
        meta = {
            "accepted": len(good),
            "rejected": {r["id"]: r["reason"] for r in results if not r["ok"]},
            "rows": [{key: v for key, v in r.items() if key not in ("pixels", "ok")} for r in good],
        }
        (chunks / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
        accepted += len(good)
        print(
            f"{name}: {len(good)}/{len(results)} accepted, total {accepted} ({time.time() - t0:.0f} s)",
            flush=True,
        )
    print("fetched", accepted)


def load_chunks(cfg: ts.TrainsetConfig) -> tuple[list[dict[str, Any]], np.ndarray, dict[int, str]]:
    """Rows and pixels of every chunk in plan order, and the rejection reasons."""
    chunks = Path(cfg.root) / "chunks"
    rows: list[dict[str, Any]] = []
    pix: list[np.ndarray] = []
    rejected: dict[int, str] = {}
    for p in sorted(chunks.glob("chunk_*.json")):
        meta = json.loads(p.read_text())
        rows.extend(meta["rows"])
        pix.append(np.load(p.with_suffix(".npz"))["pixels"])
        rejected.update({int(i): r for i, r in meta["rejected"].items()})
    return (
        rows,
        np.concatenate(pix) if pix else np.zeros((0, ts.SIZE, ts.SIZE, 3), np.uint8),
        rejected,
    )


def cmd_verify(cfg: ts.TrainsetConfig) -> None:
    ids, artists = ts.exclusions(Path(cfg.eval_manifest))
    rows, pixels, rejected = load_chunks(cfg)
    rows, pixels = (
        rows[: cfg.target],
        pixels[: cfg.target],
    )  # plan order: the surplus of the last chunk is dropped
    reference = ts.eval_hashes(Path(cfg.eval_manifest), Path(cfg.eval_prepared))
    if len(reference) < 100:
        raise SystemExit(
            f"only {len(reference)} prepared evaluation images found in {cfg.eval_prepared}: run scripts.fetch_eval_set and generate_eval_sdxl first"
        )
    hashes = {r["id"]: ts.dhash(pixels[i]) for i, r in enumerate(rows)}
    dups = ts.near_duplicates(hashes, reference, cfg.hash_distance)
    keep = [i for i, r in enumerate(rows) if r["id"] not in dups]
    rows, pixels = [rows[i] for i in keep], pixels[keep]
    check(rows, ids, artists)
    rng = random.Random(cfg.seed)
    val = set(rng.sample(sorted(r["id"] for r in rows), cfg.val_images))
    split = np.array([r["id"] in val for r in rows])
    root = Path(cfg.root)
    np.save(root / "images.npy", pixels)
    np.save(root / "object_ids.npy", np.array([r["id"] for r in rows], dtype=np.int64))
    np.save(root / "is_val.npy", split)
    (root / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    digest = ts.manifest_digest([(r["id"], r["pixel_sha256"]) for r in rows])
    ids_path = Path(cfg.ids_file)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(
        "".join(
            f"{r['id']}{' val' if r['id'] in val else ''}\n"
            for r in sorted(rows, key=lambda r: r["id"])
        ),
        encoding="utf-8",
    )
    reasons: dict[str, int] = {}
    for reason in rejected.values():
        reasons[reason.split(":")[0]] = reasons.get(reason.split(":")[0], 0) + 1
    Path(cfg.excluded_file).write_text(
        yaml.safe_dump(
            {
                "note": "evaluation objects and artists excluded from the training subset (docs/design/evaluation.md 8.2)",
                "eval_met_object_ids": sorted(ids),
                "eval_artists": sorted(artists),
                "near_duplicates_dropped": {int(k): v for k, v in sorted(dups.items())},
                "hash_distance": cfg.hash_distance,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "csv_sha256": cfg.csv_sha256,
        "n_images": len(rows),
        "n_train": int((~split).sum()),
        "n_val": int(split.sum()),
        "per_stratum": ts.class_counts(rows),
        "rejections": reasons,
        "near_duplicates_dropped": len(dups),
        "eval_images_hashed": len(reference),
        "manifest_sha256": digest,
        "pixel_mean": [float(x) for x in pixels.reshape(-1, 3).mean(0) / 255],
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


def check(rows: list[dict[str, Any]], ids: set[int], artists: set[str]) -> None:
    """Disjointness: no evaluation object, no object of an evaluation artist, no duplicate id."""
    seen = [r["id"] for r in rows]
    assert len(seen) == len(set(seen)), "duplicate object ids"
    assert not (set(seen) & ids), "an evaluation object is in the training subset"
    bad = [r["id"] for r in rows if set(ts.artist_names(r["artist"])) & artists]
    assert not bad, f"{len(bad)} objects of an evaluation artist: {bad[:5]}"


def cmd_check(cfg: ts.TrainsetConfig) -> None:
    ids, artists = ts.exclusions(Path(cfg.eval_manifest))
    root = Path(cfg.root)
    rows = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    check(rows, ids, artists)
    pixels = np.load(root / "images.npy", mmap_mode="r")
    bad = [
        r["id"]
        for i, r in enumerate(rows)
        if ts.pixel_sha256(np.asarray(pixels[i])) != r["pixel_sha256"]
    ]
    assert not bad, f"{len(bad)} images differ from the manifest checksums"
    reference = ts.eval_hashes(Path(cfg.eval_manifest), Path(cfg.eval_prepared))
    dups = ts.near_duplicates(
        {r["id"]: ts.dhash(np.asarray(pixels[i])) for i, r in enumerate(rows)},
        reference,
        cfg.hash_distance,
    )
    assert not dups, f"near duplicates of evaluation images: {dups}"
    print(
        "OK",
        len(rows),
        "images; digest",
        ts.manifest_digest([(r["id"], r["pixel_sha256"]) for r in rows]),
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=("plan", "fetch", "verify", "check"))
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(ts.TrainsetConfig, args.config, args.overrides)
    {"plan": cmd_plan, "fetch": cmd_fetch, "verify": cmd_verify, "check": cmd_check}[args.command](
        cfg
    )


if __name__ == "__main__":
    main()
