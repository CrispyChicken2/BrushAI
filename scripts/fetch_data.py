"""Download the public-domain evaluation images listed in a manifest and verify their checksums.

Usage (repo root)::

    python -m scripts.fetch_data [--manifest configs/experiments/images.yaml] [--out data/raw]
                                 [--update] [--insecure]

Files go to ``<out>/<id>.jpg`` (gitignored, never committed). If the manifest has a ``sha256``
the download is verified against it (mismatch = error); ``--update`` fills in missing checksums
(and rewrites the manifest), which is how they were recorded the first time. ``--insecure`` disables
TLS certificate verification, only for machines behind a TLS-intercepting proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

USER_AGENT = "BrushStrokeResearch/0.1 (https://github.com/CrispyChicken2/BrushAI; research)"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path, insecure: bool) -> None:
    ctx = ssl._create_unverified_context() if insecure else None  # noqa: S323
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:  # noqa: S310
        data = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def fetch(
    manifest: Path, out: Path, update: bool = False, insecure: bool = False
) -> dict[str, Path]:
    """Fetch every image of the manifest; returns id -> local path."""
    spec: dict[str, Any] = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    paths: dict[str, Path] = {}
    changed = False
    for item in spec["images"]:
        dest = out / f"{item['id']}.jpg"
        if not dest.exists() or (item["sha256"] and sha256_of(dest) != item["sha256"]):
            download(item["url"], dest, insecure)
        digest = sha256_of(dest)
        if item["sha256"] and digest != item["sha256"]:
            raise RuntimeError(
                f"{item['id']}: sha256 mismatch (expected {item['sha256']}, got {digest})"
            )
        if not item["sha256"]:
            if not update:
                raise RuntimeError(
                    f"{item['id']}: no sha256 in the manifest; run with --update once"
                )
            item["sha256"] = digest
            changed = True
        paths[item["id"]] = dest
        print(f"{item['id']}: ok ({dest}, sha256 {digest[:12]}...)")
    if changed:
        manifest.write_text(
            yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=Path("configs/experiments/images.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--update", action="store_true", help="record missing sha256 in the manifest"
    )
    parser.add_argument("--insecure", action="store_true", help="skip TLS certificate verification")
    args = parser.parse_args(argv)
    fetch(args.manifest, args.out, args.update, args.insecure)


if __name__ == "__main__":
    main(sys.argv[1:])
