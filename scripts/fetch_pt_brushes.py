"""Download the official Paint Transformer brush images to ``data/raw/pt_brush/`` and verify their
checksums.

Usage (repo root)::

    python -m scripts.fetch_pt_brushes [--insecure]

The four PNGs (Apache-2.0, from ``Huage001/PaintTransformer`` at a pinned commit) are listed with
their SHA-256 in
``configs/pt/brushes.yaml``; they are never committed. The PT-compatible renderer
(``brushstroke.renderer.pt``)
refuses to run without them (no silent fallback).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.fetch_data import fetch

MANIFEST = Path("configs/pt/brushes.yaml")
OUT = Path("data/raw/pt_brush")


def fetch_pt_brushes(insecure: bool = False, out: Path = OUT) -> dict[str, Path]:
    """Fetch and verify the PT brush images; returns asset id -> local file."""
    return fetch(MANIFEST, out, insecure=insecure, section="assets")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--insecure", action="store_true", help="skip TLS certificate verification")
    args = parser.parse_args(argv)
    fetch_pt_brushes(args.insecure)


if __name__ == "__main__":
    main(sys.argv[1:])
