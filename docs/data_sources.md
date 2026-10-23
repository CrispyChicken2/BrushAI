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

## Paint Transformer brush images and official code (T-011, T-010)

The PT-compatible renderer mode needs the four brush images of the official repository `Huage001/PaintTransformer`
(Apache License 2.0, commit `7960b8eef2f6af7171af01933a3d8dbc73190a8d`; attribution: Songhua Liu et al., Baidu). They are
**fetched, never committed**, by `python -m scripts.fetch_pt_brushes [--insecure]` (manifest `configs/pt/brushes.yaml`, SHA-256 verified):
the `train_*` pair is what `train/models/painter_model.py` reads, the `inference_*` pair what `inference/inference.py` reads (different files).
If they are missing the PT mode raises an error that names this command; there is no fallback brush.

| asset id | file in `data/raw/pt_brush/` | source (commit 7960b8e) | license | SHA-256 |
|---|---|---|---|---|
| `inference_brush_large_vertical` | `inference_brush_large_vertical.png` | <https://raw.githubusercontent.com/Huage001/PaintTransformer/7960b8eef2f6af7171af01933a3d8dbc73190a8d/inference/brush/brush_large_vertical.png> | Apache-2.0 | `3b2074c1b1f0b32b1b049a3810de544acf61553196d8cf0ce5458a3f64dbc98a` |
| `inference_brush_large_horizontal` | `inference_brush_large_horizontal.png` | <https://raw.githubusercontent.com/Huage001/PaintTransformer/7960b8eef2f6af7171af01933a3d8dbc73190a8d/inference/brush/brush_large_horizontal.png> | Apache-2.0 | `d05607378b8ce2efe55a1d35c99d85f9b87b10f696f9481f727acca700ae5bb9` |
| `train_brush_large_vertical` | `train_brush_large_vertical.png` | <https://raw.githubusercontent.com/Huage001/PaintTransformer/7960b8eef2f6af7171af01933a3d8dbc73190a8d/train/brush/brush_large_vertical.png> | Apache-2.0 | `7a59df5d865cd7688a44d2e0d7c14b01d8c676178f4e1121e1c19b1f9d1f61c1` |
| `train_brush_large_horizontal` | `train_brush_large_horizontal.png` | <https://raw.githubusercontent.com/Huage001/PaintTransformer/7960b8eef2f6af7171af01933a3d8dbc73190a8d/train/brush/brush_large_horizontal.png> | Apache-2.0 | `9fc5dc4d7d968ad50a1e6b5cfb529e863354843d7c1ff6c8b9547e658d242779` |

The official code files used for the parity tests and the pretrained weights are fetched by `python -m scripts.pt_reference_study fetch`
into `data/raw/pt_official/` (also gitignored); see `docs/design/paint_transformer.md` section 8 for their license status (the weights
carry no separate license: reference use only, never redistributed).

## Models / weights (not redistributed)
- VGG16 (ImageNet) backbone of LPIPS-VGG and AlexNet (ImageNet) of LPIPS-Alex: downloaded by `torchvision` into the torch
  hub cache on first use. LPIPS linear heads: shipped with the `lpips` package (BSD-2-Clause, Zhang et al.).

## Paint Transformer code and pretrained weights (T-010, reference only)
Used only as a reference point, outside git (`scripts/pt_reference_study.py`; details and citations in
[design/paint_transformer.md](design/paint_transformer.md), section 8). The brush images are listed above.
- **Code**: `Huage001/PaintTransformer` (the author's PyTorch re-implementation), Apache License 2.0, pinned to commit
  `7960b8eef2f6af7171af01933a3d8dbc73190a8d`; each file is checked against a SHA-256 in the script. Downloaded to
  `data/raw/pt_official/` (gitignored). No files of the repository are copied into this one.
- **Pretrained weights** (`model.pth`, 36.3 MB, Google Drive link from the README, SHA-256 `1fbd761a...`, full value in the script):
  no license is stated by the authors. Treated as research reference material: never committed or redistributed, loaded with
  `torch.load(..., weights_only=True)`.
- The paper (Liu et al., ICCV 2021, arXiv 2108.03798) is cited, not copied.
- The datasets of the paper (Kaggle landscape pictures, WikiArt, FFHQ) are not used.
