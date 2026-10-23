# Data sources

Everything used by the experiments is public domain or CC0 and is **downloaded, never committed**
(`data/raw/` is gitignored). Checksums are verified by the fetch script.

## Evaluation images (T-009 sweep)

Fetched by `python -m scripts.fetch_data` from the manifest `configs/experiments/images.yaml` (add `--insecure` behind a
TLS-intercepting proxy). Each file is the 960 px thumbnail served by Wikimedia Commons; the license strings are those of the
Commons file pages (checked through the Commons API on 2026-10-22). The paintings are two-dimensional reproductions of works
whose authors died more than 70 years ago (public domain in their countries of origin and the US); the Apollo 17 photograph is a
NASA work (US federal government, public domain).

| id | work | author | date | source page | license | SHA-256 (960 px thumbnail) |
|----|------|--------|------|-------------|---------|---------|
| `mona_lisa` | Mona Lisa | Leonardo da Vinci | c. 1503-1506 | <https://commons.wikimedia.org/wiki/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg> | Public domain (2-D reproduction of a work whose author died in 1519) | `5f223c31ba0a477eb7cbe5e5f959d822cbfc46081b19b96498cbd5601d9ac81d` |
| `starry_night` | The Starry Night | Vincent van Gogh | 1889 | <https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg> | Public domain (author died in 1890) | `70ddc5f0b95875823ddf261ad749b0faaaa5405b5cfcfdd803628c28da847137` |
| `great_wave` | The Great Wave off Kanagawa (Metropolitan Museum impression) | Katsushika Hokusai | c. 1830-1832 | <https://commons.wikimedia.org/wiki/File:Tsunami_by_hokusai_19th_century.jpg> | Public domain (author died in 1849) | `f615e079c64a69bcb50bd6c832c9d5d35359077c216ed90bf5806ef28df01c0c` |
| `pearl_earring` | Girl with a Pearl Earring | Johannes Vermeer | c. 1665 | <https://commons.wikimedia.org/wiki/File:Meisje_met_de_parel.jpg> | Public domain (author died in 1675) | `e9b3dca43a5c0eda5e6cd0aca4f24f1f24c9b2d0dd25695158dc5d48d5b39408` |
| `earth_apollo17` | The Earth seen from Apollo 17 ("Blue Marble") | NASA / Apollo 17 crew | 1972-12-07 | <https://commons.wikimedia.org/wiki/File:The_Earth_seen_from_Apollo_17.jpg> | Public domain (NASA work) | `16eaa54446f761a80ee8de604107b66432a1e6b489aa99d4777dc92765fae80a` |

Images are center-cropped to a square and resized to 512 x 512 before painting (`brushstroke.utils.io.center_crop_resize`).

## Demo inputs (T-007, M1 demo GIFs)

The three M1 demos reuse images of the table above (same manifest, same checksums; no second downloader). They are downloaded to
`data/raw/demo/` (gitignored) by `python -m scripts.fetch_demo_inputs [--insecure]` (also run by `scripts/make_m1_demos.py`), which verifies
the SHA-256 of every file against `configs/experiments/images.yaml`. The demo GIFs in `docs/media/` are derivative works (paintings
generated from these public-domain images); the inputs themselves are never committed.

| demo | role | image id | work | author | date | source page | license | SHA-256 |
|------|------|----------|------|--------|------|-------------|---------|---------|
| `m1_portrait` | portrait / figure | `pearl_earring` | Girl with a Pearl Earring | Johannes Vermeer | c. 1665 | <https://commons.wikimedia.org/wiki/File:Meisje_met_de_parel.jpg> | Public domain (author died in 1675) | `e9b3dca43a5c0eda5e6cd0aca4f24f1f24c9b2d0dd25695158dc5d48d5b39408` |
| `m1_landscape` | landscape | `starry_night` | The Starry Night | Vincent van Gogh | 1889 | <https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg> | Public domain (author died in 1890) | `70ddc5f0b95875823ddf261ad749b0faaaa5405b5cfcfdd803628c28da847137` |
| `m1_photo` | photo with a clear subject | `earth_apollo17` | The Earth seen from Apollo 17 ("Blue Marble") | NASA / Apollo 17 crew | 1972-12-07 | <https://commons.wikimedia.org/wiki/File:The_Earth_seen_from_Apollo_17.jpg> | Public domain (NASA work) | `16eaa54446f761a80ee8de604107b66432a1e6b489aa99d4777dc92765fae80a` |

## Models / weights (not redistributed)
- VGG16 (ImageNet) backbone of LPIPS-VGG and AlexNet (ImageNet) of LPIPS-Alex: downloaded by `torchvision` into the torch
  hub cache on first use. LPIPS linear heads: shipped with the `lpips` package (BSD-2-Clause, Zhang et al.).
