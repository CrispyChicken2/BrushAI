"""Fetch the evaluation set (T-024) from its manifest and verify every checksum.

    python -m scripts.fetch_eval_set [--out data/raw/eval] [--check] [--insecure] [--sdxl]

For each Met / Wikimedia Commons image of ``configs/eval/eval_set.yaml``: download the source file to
``<out>/source/<id>.<ext>`` unless it is already there, prepare the 512 x 512 image (centre crop or the manifest
``crop_box``, box filter) and check the SHA-256 of its pixels against the manifest: a pixel mismatch is an error, never
silently accepted. The SHA-256 of the file is compared too; a museum can re-encode a file without changing a pixel, so a
file mismatch with identical pixels is reported ("file bytes differ") and accepted. Then write ``<out>/prepared/<id>.png``. The 30 SDXL images are not downloaded:
``--sdxl`` runs ``python -m scripts.generate_eval_sdxl`` (which verifies their hashes); without it they are only verified
if they already exist. ``--check`` downloads nothing and reports what is missing or wrong. The files are never committed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image

from brushstroke.eval.dataset import (
    EvalImage,
    load_manifest,
    pixel_sha256,
    prepare_image,
    sha256_file,
    validate_manifest,
)
from brushstroke.utils.http import Http

MANIFEST = Path("configs/eval/eval_set.yaml")


def source_path(out: Path, im: EvalImage) -> Path:
    ext = Path(urllib.parse.urlparse(im.url).path).suffix.lower() or ".jpg"
    return out / "source" / f"{im.id}{ext}"


def fetch_one(im: EvalImage, out: Path, http: Http | None) -> str:
    """Download (if allowed and needed), verify and prepare one image; returns a status string."""
    src = source_path(out, im)
    if not src.exists():
        if http is None:
            return "missing"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(http.get(im.url))
    same_bytes = sha256_file(src) == im.sha256
    try:
        with Image.open(src) as pil:
            pixels = prepare_image(pil, im.crop_box)
    except (
        ValueError
    ) as err:  # the file is not the one of the manifest (another size, a crop box outside it)
        return f"SHA-256 MISMATCH (source file{'' if same_bytes else ' changed'}): {err}"
    if pixel_sha256(pixels) != im.prepared_sha256:
        return "SHA-256 MISMATCH (prepared pixels)" + ("" if same_bytes else " and source file")
    # A museum can re-encode a file (metadata, container) without changing a pixel: the pixels decide.
    status = (
        "ok" if same_bytes else "ok (file bytes differ from the manifest, decoded pixels identical)"
    )
    dest = out / "prepared" / f"{im.id}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(dest)
    return status


def verify_sdxl(im: EvalImage, out: Path) -> str:
    path = out / "sdxl" / f"{im.id}.png"
    prepared = out / "prepared" / f"{im.id}.png"
    if not path.exists() or not prepared.exists():
        return "missing (run python -m scripts.generate_eval_sdxl)"
    ok = pixel_sha256(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)) == im.sha256
    ok &= (
        pixel_sha256(np.asarray(Image.open(prepared).convert("RGB"), dtype=np.uint8))
        == im.prepared_sha256
    )
    return "ok" if ok else "SHA-256 MISMATCH (GPU nondeterminism? see the data card)"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, default=Path("data/raw/eval"))
    parser.add_argument("--check", action="store_true", help="download nothing, report")
    parser.add_argument("--sdxl", action="store_true", help="also (re)generate the SDXL images")
    parser.add_argument("--insecure", action="store_true", help="skip TLS certificate verification")
    args = parser.parse_args(argv)

    header, images = load_manifest(args.manifest)
    problems = validate_manifest(header, images)
    if problems:
        raise SystemExit("invalid manifest:\n  " + "\n  ".join(problems))
    http = None if args.check else Http(args.insecure)
    bad_files = bad_sdxl = (
        0  # the failures of the downloaded images are never masked by the SDXL result
    )
    for im in images:
        is_sdxl = im.category == "sdxl"
        status = verify_sdxl(im, args.out) if is_sdxl else fetch_one(im, args.out, http)
        ok = status.startswith("ok")
        bad_sdxl += is_sdxl and not ok
        bad_files += not is_sdxl and not ok
        print(f"{im.id}: {status}")
    if args.sdxl and not args.check:
        subprocess.run(
            [sys.executable, "-m", "scripts.generate_eval_sdxl", "--out", str(args.out)], check=True
        )
        bad_sdxl = sum(verify_sdxl(im, args.out) != "ok" for im in images if im.category == "sdxl")
    bad = bad_files + bad_sdxl
    print(f"{len(images)} images, {bad} not ok")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
