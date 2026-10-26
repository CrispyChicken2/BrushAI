"""Fetch the InSPyReNet evaluation saliency model (T-025): code at a pinned commit and the DUTS-TR SwinB checkpoint.

    python -m scripts.fetch_inspyrenet [--root data/raw/inspyrenet] [--check] [--insecure]

The pins are in ``configs/eval/saliency.yaml``: the files of ``github.com/plemeri/InSPyReNet`` at one commit (MIT), each
with its SHA-256, and the checkpoint ``InSPyReNet_SwinB`` trained on DUTS-TR (Google Drive file of the model zoo) with its
SHA-256. A file that does not match its hash is an error and is deleted. Nothing is committed: ``data/raw`` is git-ignored
and the checkpoint, which carries no licence statement, is never redistributed. The Google Drive download needs ``gdown``
(``pip install gdown``); behind a TLS-intercepting proxy the system certificates are used through ``truststore`` if it is installed,
and ``--insecure`` skips the verification of the certificate as a last resort.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from brushstroke.eval.saliency import ROOT, load_spec, sha256_file, verify_files
from brushstroke.utils.http import Http

RAW = "https://raw.githubusercontent.com/plemeri/InSPyReNet/{commit}/{path}"


def fetch_code(root: Path, spec: dict, http: Http) -> None:
    for rel, want in spec["code"]["files"].items():
        dest = root / "code" / rel
        if dest.exists() and sha256_file(dest) == want:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(http.get(RAW.format(commit=spec["code"]["commit"], path=rel)))
        if sha256_file(dest) != want:
            dest.unlink()
            raise SystemExit(f"{rel}: SHA-256 differs from the pinned one (deleted)")
        print(f"{rel}: ok")


def fetch_checkpoint(root: Path, spec: dict, insecure: bool) -> None:
    ckpt = spec["checkpoint"]
    dest = root / ckpt["file"]
    if dest.exists() and sha256_file(dest) == ckpt["sha256"]:
        return
    try:
        import gdown
    except ImportError as err:
        raise SystemExit("the checkpoint comes from Google Drive: pip install gdown") from err
    try:  # use the system certificate store (TLS-intercepting proxies)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    root.mkdir(parents=True, exist_ok=True)
    gdown.download(
        id=ckpt["sources"]["google_drive_id"], output=str(dest), quiet=False, verify=not insecure
    )
    if not dest.exists() or sha256_file(dest) != ckpt["sha256"]:
        dest.unlink(missing_ok=True)
        raise SystemExit(
            f"{ckpt['file']}: missing or SHA-256 differs from the pinned one (deleted)"
        )
    print(f"{ckpt['file']}: ok")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--check", action="store_true", help="verify the files on disk, download nothing"
    )
    parser.add_argument("--insecure", action="store_true", help="skip TLS certificate verification")
    args = parser.parse_args(argv)
    spec = load_spec()
    if not args.check:
        fetch_code(args.root, spec, Http(args.insecure))
        fetch_checkpoint(args.root, spec, args.insecure)
    problems = verify_files(args.root, spec)
    for p in problems:
        print("PROBLEM", p)
    print("InSPyReNet files: " + ("ok" if not problems else f"{len(problems)} problem(s)"))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
