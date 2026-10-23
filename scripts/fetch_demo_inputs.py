"""Download the three M1 demo inputs to ``data/raw/demo/`` and verify their checksums.

Usage (repo root)::

    python -m scripts.fetch_demo_inputs [--insecure]

A thin wrapper over :func:`scripts.fetch_data.fetch`: the inputs are the images of
``configs/experiments/images.yaml`` named by ``image_id`` in ``configs/demos/m1_*.yaml``
(sources, licenses and SHA-256 in ``docs/data_sources.md``). Never committed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from scripts.fetch_data import fetch

DEMO_DIR = Path("configs/demos")
MANIFEST = Path("configs/experiments/images.yaml")
OUT = Path("data/raw/demo")


def demo_image_ids(demo_dir: Path = DEMO_DIR) -> list[str]:
    """``image_id`` of every ``m1_*.yaml`` demo config, in file-name order."""
    ids = []
    for path in sorted(demo_dir.glob("m1_*.yaml")):
        ids.append(yaml.safe_load(path.read_text(encoding="utf-8"))["image_id"])
    return ids


def fetch_demo_inputs(insecure: bool = False, out: Path = OUT) -> dict[str, Path]:
    """Fetch (and verify) the demo inputs; returns image id -> local file."""
    return fetch(MANIFEST, out, insecure=insecure, ids=demo_image_ids())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--insecure", action="store_true", help="skip TLS certificate verification")
    args = parser.parse_args(argv)
    fetch_demo_inputs(args.insecure)


if __name__ == "__main__":
    main(sys.argv[1:])
