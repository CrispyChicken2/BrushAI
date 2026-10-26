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

## Evaluation saliency model (T-025, M3)

The saliency-weighted LPIPS and the saliency control of `docs/design/evaluation.md` section 4 use **InSPyReNet** (Kim et al., ACCV 2022), SwinB backbone, trained on DUTS-TR only, as a measuring instrument. Pins in `configs/eval/saliency.yaml`; one command fetches and verifies them: `python -m scripts.fetch_inspyrenet` (needs `requirements-eval.txt`).

| item | source | licence | pin |
|---|---|---|---|
| code (15 files: `lib/`, `utils/misc.py`, config, `LICENSE`) | <https://github.com/plemeri/InSPyReNet> at commit `692ad8a5774cf95ef615a730d5070aad18f6b160` | MIT (Copyright 2021 Taehun Kim; GitHub licence API: MIT) | SHA-256 of every file |
| checkpoint `InSPyReNet_SwinB`, DUTS-TR (367,526,592 bytes, 90.7 M parameters) | model zoo of the repository (`docs/model_zoo.md`, row SwinB / DUTS-TR), Google Drive file `1k5hNJImgEgSmz-ZeJEEb_dVkrOnswVMq` (the OneDrive mirror answered 403) | **none stated** for the checkpoints (checked 2026-10-26: README, model zoo page and `LICENSE` cover the code) | SHA-256 `6a35253650457b2d87d8704b08e19d90aef58da081456c18ac77cd357addfaee` |

The checkpoint is treated like the Paint Transformer weights: fetched into `data/raw/inspyrenet/` (git-ignored), hash pinned, used for measuring only, **never committed or redistributed**. The training set DUTS-TR is not used here. The maps of the 100 evaluation images are derived data: cached in `outputs/saliency/` (git-ignored, regenerated by `python -m scripts.compute_saliency`), with the SHA-256 of every map recorded in `docs/experiments/data/saliency/saliency_maps.json`.

The control, the spectral residual of Hou and Zhang (CVPR 2007), needs no weights and no licence. **InSPyReNet is reserved for evaluation: the contribution (M4) must not use it** (circularity rule, `evaluation.md` 4.2; `tests/test_circularity.py` fails if a method package or a non-saliency script mentions it).

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

## Evaluation set (T-024, M3)

The 100 images of the evaluation protocol (`docs/design/evaluation.md` section 1): 40 paintings, 30 photographs, 30 SDXL images. The manifest
`configs/eval/eval_set.yaml` lists every image with its source page and file URL, title, author, date, licence string, retrieval date,
SHA-256 of the file used, SHA-256 of the prepared 512 x 512 pixels, and the dev flag; the images themselves are **fetched, never
committed** (`data/raw/eval/`, gitignored). One command each:

```
python -m scripts.fetch_eval_set            # download + verify the 70 Met / Commons images, write data/raw/eval/prepared/*.png
python -m scripts.generate_eval_sdxl        # generate the 30 SDXL images (GPU, about 10 min) and verify their hashes
python -m scripts.fetch_eval_set --check    # verify everything that is on disk, download nothing
python -m scripts.build_eval_set check      # no image of the set equals / resembles a tuning image (writes overlap_check.json)
python -m scripts.audit_eval_set            # every machine-checkable rule on every image, 0 violations expected (writes audit.json)
```

The rules that need eyes were applied to every image on contact sheets and are recorded one line per image in
`docs/experiments/data/eval_set/visual_audit.md` (T-034).

**Sources and licences (checked 2026-10-26).**
- *Paintings (40)*: Metropolitan Museum Open Access, `isPublicDomain = true`, CC0 (API v1.1, object list per department; the `search` endpoint
  was retired on 2026-10-01). 24 European paintings (8 portraits / 8 landscapes / 8 other, from the Met subject tags), 8 Asian (4 paintings,
  4 woodblock prints), 8 other (4 Modern Art, 4 American Wing).
- *Photographs (30)*: 9 nineteenth-century portrait photographs of the Met (CC0, monochrome or sepia prints, one hand-coloured, sitters long dead); 21 colour
  photographs from Wikimedia Commons whose file page carries a public-domain tag and whose source is a US federal collection: 14 Library of
  Congress FSA/OWI Kodachrome slides (1939-1943, US government work), 7 US Fish and Wildlife Service wildlife photographs (animals only: a species
  category and the animal in the title, no word about people, staff or events, and no "person" found by a COCO detector, torchvision Faster R-CNN MobileNetV3, checked again
  on the final set). Prokudin-Gorsky plates were tried and dropped: most are sepia-toned reproductions, not colour photographs. 10 portraits (9 + 1), 7 animals, 6 objects, 7 scenes. Smithsonian Open
  Access was not needed and is not used, so its per-file statement was not relied on. Excluded: any image of a person who may be alive
  (no child and no possibly living identifiable person among the FSA or agency photographs: title rule, people detector for the agency photographs, and the image-by-image visual audit; the Met portraits are long-dead historical sitters of any age, one of them a girl; the adults of 1940-43 slides cannot be alive).
  `colour: false` marks the 8 monochrome or sepia Met prints; the hand-coloured Met portrait is `colour: true`.
- *SDXL (30)*: Stable Diffusion XL base 1.0, revision `462165984030d82259a11f4367a4eed129e94a7b`, licence CreativeML Open RAIL++-M (Licensor
  claims no rights in the outputs; the use restrictions do not concern the prompts). The 30 outputs are dedicated to CC0 1.0. Settings are
  the library defaults (`configs/eval/sdxl_prompts.yaml`, dependencies pinned in `requirements-gen.txt`); fp16 weights with model CPU offload; `diffusers` 0.42.0, `torch` 2.11.0+cu128.
  Measured on the RTX 5070: peak VRAM 5.3 GB allocated / 7.5 GB reserved, 20 to 32 s per image (20 s alone, 32 s with the CPU busy).

**Selection (all mechanical, none looks at a method result).** `random.Random(22)`; per department or category the candidate ids sorted then
shuffled; the first candidates that pass the rules are kept (`scripts/build_eval_set.py`, docstring). Rules: public domain / CC0 with a
per-file statement, source short side >= 768 px and aspect in [3:4, 4:3], colour for the Commons photographs, at most 4 images per
photographer (3 until review round 2: once the sepia Prokudin plates were out, the FSA pool could not fill the strata at 3; names normalized: "Lee, Russell, 1903-1986, photographer" and "Russell Lee" count together), no two views of one scene
(same photographer and a shared specific category such as a place, or similar titles; applied to the FSA photographs, not to agency wildlife
photographs), no near-duplicate of an accepted or of a tuning image. The 52 visual rejections taken on the contact sheets (text over 5 %
of the area, frames and mats, no clear subject, possible living person, sepia tint, duplicates) are listed with their reason in
`configs/eval/curation_rejections.yaml`, and every rejected candidate was replaced by the next one of the seeded draw. Preparation: centre crop
(40 images carry a mechanical `crop_box` that removes uniform black film or mat borders, side >= 640 px) and a Pillow BOX (area) reduction to 512.

**Reproducibility, measured.**
- Fetch: a download from scratch into an empty directory reproduced the prepared pixels of all 70 images (SHA-256) in 64 s. One Met file
  (`met_488147`) had changed *bytes* between the curation and the check (the museum re-encoded it, same decoded pixels; a second one did in the first draw), so the
  verification decides on the prepared pixels and reports the file difference. The curation replay (`python -m scripts.build_eval_set curate`) writes a byte-identical manifest.
- SDXL: two complete generations on the same machine are bitwise identical (30 of 30 images, max difference 0), and a third run verified all
  the hashes of the manifest. GPU kernels are not guaranteed bitwise stable across drivers and devices: `generate_eval_sdxl` reports the pixel
  difference of a mismatch (`--allow-mismatch` to continue); the hashes in the manifest are those of this machine (record: `docs/experiments/data/eval_set/sdxl_generation.json`).
- Overlap with tuning: the six images of the T-009 sweep and the M1/M2 demos (Mona Lisa, Starry Night, Great Wave, Pearl Earring, Earth from
  Apollo 17, Kirkjufell) are not in the set; the largest 32 x 32 thumbnail correlation between any set image and any of them is 0.59
  (threshold 0.9); the Met impression of the Great Wave (object 45434) was excluded from the draw by id. Two SDXL prompts fixed in T-022 are
  pastiches of tuning images (`sdxl_19`: waves and a distant mountain in woodblock style, `sdxl_18`: a swirling sky over sunflowers); they are test images,
  not copies, and the thumbnail check does not flag them (see the correlation above), but a reader should know the resemblance of motif. Record: `docs/experiments/data/eval_set/overlap_check.json`.

**Dev split (10, fixed here before any method ran; redrawn in review rounds 1 to 3 and in T-034 when test-set defects were corrected: no method had run).** 4 paintings (2 European, 1 Asian, 1 other), 3 photographs (1 portrait, 2 others), SDXL
prompts 04, 14, 25. Drawn with the same `random.Random(22)` after the three draws, flagged `dev: true` in the manifest and in the table below.

**Limits.** The photographs are dominated by one US federal collection (FSA/OWI 1939-1943) and the portraits by 19th-century prints, so they
are not representative of modern photography; the paintings are mostly European; after the trimming (which sees through a thin light edge line, with one definition of a border for the trimming and the report), the border of any image is at most 4.5 % of the area (measured by `audit_eval_set`; a slide with a film bar next to a dark sky, and three slides with a light line in front of the bar, carried 6 % or more before and were rejected or trimmed); inscriptions that are part of an East Asian work are kept (below a third of the area).

<!-- BEGIN EVALSET -->

| category | stratum | n | dev | licence |
|---|---|---|---|---|
| painting | asian | 8 | 1 | CC0 1.0 (Met Open Access, isPublicDomain) |
| painting | european_landscape | 8 | 0 | CC0 1.0 (Met Open Access, isPublicDomain) |
| painting | european_other | 8 | 2 | CC0 1.0 (Met Open Access, isPublicDomain) |
| painting | european_portrait | 8 | 0 | CC0 1.0 (Met Open Access, isPublicDomain) |
| painting | other_art | 8 | 1 | CC0 1.0 (Met Open Access, isPublicDomain) |
| photo | photo_animal | 7 | 1 | Public domain (US government work, US Fish and Wildlife Service) |
| photo | photo_object | 6 | 0 | Public domain (US government work, Library of Congress FSA/OWI) |
| photo | photo_portrait | 9 | 1 | CC0 1.0 (Met Open Access, isPublicDomain) |
| photo | photo_portrait | 1 |  | Public domain (US government work, Library of Congress FSA/OWI) |
| photo | photo_scene | 7 | 1 | Public domain (US government work, Library of Congress FSA/OWI) |
| sdxl | sdxl_object | 10 | 1 | CC0 1.0 (own-generated, SDXL output) |
| sdxl | sdxl_portrait | 10 | 1 | CC0 1.0 (own-generated, SDXL output) |
| sdxl | sdxl_scene | 10 | 1 | CC0 1.0 (own-generated, SDXL output) |

| id | category / stratum | title | author | date | dev |
|---|---|---|---|---|---|
| `met_695496` | painting / european_portrait | Bust of a Man (Saint Matthias?) | Giovanni Battista Piazzetta | ca. 1715–20 |  |
| `met_435862` | painting / european_other | The Route Nationale at Samer | Jean-Charles Cazin |  |  |
| `met_748947` | painting / european_portrait | Ellen Maurice (1578–1626) | Marcus Gheeraerts the Younger | 1597 |  |
| `met_436642` | painting / european_other | The Musician | Bartholomeus van der Helst | 1662 |  |
| `met_436262` | painting / european_other | Virgin and Child with Saint Catherine of Alexandria | Anthony van Dyck | ca. 1630 | dev |
| `met_439977` | painting / european_other | Christ Carrying the Cross | Jan Gossart (called Mabuse) | ca. 1520–25 |  |
| `met_436496` | painting / european_other | Saint John the Evangelist with Acteus and Eugenius | Francescuccio Ghissi (Francesco di Cecco | ca. 1370 |  |
| `met_437921` | painting / european_portrait | Portrait of a Man | Abraham de Vries | 1643 |  |
| `met_438545` | painting / european_other | Léon Pallière (1787–1820) in His Room at the Villa Medici, R | Jean Alaux | 1817 | dev |
| `met_436616` | painting / european_other | Young Man and Woman in an Inn | Frans Hals | 1623 |  |
| `met_437897` | painting / european_other | Abraham's Parting from the Family of Lot | Jan Victors | ca. 1655–65 |  |
| `met_436824` | painting / european_portrait | Lady Mary Berkeley | Sir Godfrey Kneller | ca. 1700 |  |
| `met_437990` | painting / european_portrait | Seated Peasant | Paul Cézanne | ca. 1892–96 |  |
| `met_435993` | painting / european_portrait | Marianne Dorothy Harland (1759–1785), Later Mrs. William Dal | Richard Cosway |  |  |
| `met_436253` | painting / european_portrait | Lucas van Uffel (died 1637) | Anthony van Dyck | ca. 1622 |  |
| `met_437386` | painting / european_portrait | Portrait of a Man, probably a Member of the Van Beresteyn Fa | Rembrandt (Rembrandt van Rijn) | 1632 |  |
| `met_437211` | painting / european_landscape | The Rest on the Flight into Egypt | Netherlandish (Antwerp) Painters |  |  |
| `met_436025` | painting / european_landscape | View of Ornans | Gustave Courbet | probably mid-1850s |  |
| `met_437179` | painting / european_landscape | Near Penshurst, Kent | Patrick Nasmyth | 1828 |  |
| `met_436782` | painting / european_landscape | Capriccio with St. Paul's and Old London Bridge | Antonio Joli | ca. 1745 |  |
| `met_436183` | painting / european_landscape | Departure of the Amazons | Claude Déruet | 1620s |  |
| `met_436830` | painting / european_landscape | A Panoramic Landscape with a Country Estate | Philips Koninck | ca. 1649 |  |
| `met_436971` | painting / european_landscape | The Flood Gate | Emile van Marcke |  |  |
| `met_435880` | painting / european_landscape | Rocks at Fontainebleau | Paul Cézanne | 1890s |  |
| `met_72416` | painting / asian | Landscape | Watanabe Shikō 渡辺始興 | first half of the 18th century |  |
| `met_54318` | painting / asian | Kanjo (Court Lady) Seated, and a Tsurikoro Hanging near Her  | Totoya Hokkei | ca. 1825 |  |
| `met_54196` | painting / asian | Snowy Landscape | Ishikawa Kazan | ca. 1820 | dev |
| `met_54675` | painting / asian | Bird Catching Fish Among Reeds | Watanabe Seitei | ca. 1887 |  |
| `met_48980` | painting / asian | Spring and Summer Flowers with Bamboo | unknown | 18th century |  |
| `met_54224` | painting / asian | Print | Totoya Hokkei | 1780–1850 |  |
| `met_54923` | painting / asian | Nanamaro and His Followers Looking at the Moon in China | Kubo Shunman | 19th century |  |
| `met_74670` | painting / asian | A Lion Fighting a Crocodile | unknown | early 19th century |  |
| `met_488147` | painting / other_art | Still Life | Ben Benn | 1941 |  |
| `met_488126` | painting / other_art | "One Third of a Nation" | O. Louis Guglielmi | 1939 |  |
| `met_485551` | painting / other_art | Self-Portrait | Umberto Boccioni | 1905 |  |
| `met_488903` | painting / other_art | Jeanne Hébuterne | Amedeo Modigliani | 1919 |  |
| `met_11718` | painting / other_art | Samuel Mifflin | Charles Willson Peale | 1777–80 |  |
| `met_701989` | painting / other_art | Emma Van Name | Joshua Johnson | ca. 1805 |  |
| `met_11241` | painting / other_art | Augustus Washington Clason | John Wesley Jarvis | ca. 1820 |  |
| `met_13102` | painting / other_art | Portrait of a Lady (Possibly of the Buloid Family) | Samuel Lovett Waldo | 1832 | dev |
| `met_283187` | photo / photo_portrait | [Senator and Mrs. James Henry Lane] | Brady & Co. | 1861–66 | dev |
| `met_285903` | photo / photo_portrait | Portrait of His Daughter Reading | Louis-Désiré Blanquart-Évrard | ca. 1847 |  |
| `met_261532` | photo / photo_portrait | Marie Stuart | Pierre-Louis Pierson | 1860s |  |
| `met_263532` | photo / photo_portrait | [Performer in Bugaku-style Costume] | Unknown | 1880s |  |
| `met_268534` | photo / photo_portrait | Cunningham, Beff, John Hamilton, Guthrie | Hill and Adamson | 1843–47 |  |
| `met_261324` | photo / photo_portrait | Derelitta | Pierre-Louis Pierson | 1860s |  |
| `met_268532` | photo / photo_portrait | Dr. Welsh | Hill and Adamson | 1843–47 |  |
| `met_285608` | photo / photo_portrait | Scherzo di Follia | Pierre-Louis Pierson | 1861–67, printed ca. 1930 |  |
| `met_286569` | photo / photo_portrait | Omar Pasha | Roger Fenton | 1855 |  |
| `commons_guide_at_little_norway_blue_mounds_wis_lccn2017878934` | photo / photo_portrait | Guide at Little Norway, Blue Mounds, Wis. LCCN2017878934 | Arthur Rothstein | 1942-08 |  |
| `commons_fox_squirrell_51384470246` | photo / photo_animal | Fox Squirrell (51384470246) | USFWS Mountain Prairie | 10 June 2021, 12:04:26 |  |
| `commons_dehydrated_key_deer_36952540200` | photo / photo_animal | Dehydrated key deer (36952540200) | U.S. Fish and Wildlife Service Southeast | 2017-09-20 13:24 |  |
| `commons_trumpeter_swan_at_seedskadee_national_wildlife_refuge_497247` | photo / photo_animal | Trumpeter swan at Seedskadee National Wildlife Refuge (49724 | USFWS Mountain Prairie | 18 March 2020, 16:16:59 |  |
| `commons_short_eared_owl_at_funk_waterfowl_production_area_4100881413` | photo / photo_animal | Short-eared Owl at Funk Waterfowl Production Area (410088141 | USFWS Mountain Prairie | 28 December 2017, 13:23:34 |  |
| `commons_ruddy_duck_scott_somershoe_49717209451` | photo / photo_animal | Ruddy Duck, Scott Somershoe (49717209451) | USFWS Mountain Prairie | 25 October 2019, 09:16:35 | dev |
| `commons_white_tailed_deer_53627568015` | photo / photo_animal | White-tailed deer (53627568015) | U.S. Fish and Wildlife Service - Midwest | 29 March 2024, 19:10:07 |  |
| `commons_wild_turkey_52870186395` | photo / photo_animal | Wild turkey (52870186395) | U.S. Fish and Wildlife Service - Midwest | 4 May 2023, 08:53:37 |  |
| `commons_harrowing_a_field_with_a_diesel_tractor_seabrook_farm_bridge` | photo / photo_object | Harrowing a field with a diesel tractor, Seabrook Farm, Brid | Collier, John, Jr., 1913-1992, photograp | 1942-06 |  |
| `commons_street_in_san_juan_1a34067v` | photo / photo_object | Street in San Juan 1a34067v | Jack Delano | 1941-12 |  |
| `commons_door_and_window_in_a_spanish_american_home_1a34162v` | photo / photo_object | Door and window in a Spanish-American home 1a34162v | Russell Lee | July 1940 |  |
| `commons_laundry_barbershop_and_stores_washington_d_c_1a34426v` | photo / photo_object | [Laundry, barbershop and stores, Washington, D.C.?] | Louise Rosskam | between 1941 and 1942 |  |
| `commons_russellllanodesanjuan` | photo / photo_object | RussellLLanodeSanJuan | Russell Lee | between July 1940 and October 1940 |  |
| `commons_seed_and_feed_store_lincoln_nebraska1a34278v` | photo / photo_object | Seed and feed store, Lincoln Nebraska | John Vachon | 1942 |  |
| `commons_near_white_plains_ga_lccn2017877524` | photo / photo_scene | Near White Plains, Ga. LCCN2017877524 | Jack Delano | 1941 |  |
| `commons_a_street_in_a_town_of_the_virgin_islands1a33999v` | photo / photo_scene | A street in a town of the Virgin Islands, Christiansted, Sai | Jack Delano | December 1941 |  |
| `commons_scrap_and_salvage_depot_butte_mont_lccn2017878813` | photo / photo_scene | Scrap and salvage depot, Butte, Mont. LCCN2017878813 | Russell Lee | 1942-10 |  |
| `commons_russell_lee_street_corner_dillon_montana` | photo / photo_scene | Russell Lee street corner Dillon Montana | Russell Lee | 1942-08 |  |
| `commons_the_greensboro_lumber_co_greensboro_ga_lccn2017877531` | photo / photo_scene | The Greensboro Lumber Co., Greensboro, Ga. LCCN2017877531 | Jack Delano | 1941-06 |  |
| `commons_traffic_at_5_30_on_second_avenue_detroit_mich_lccn2017878611` | photo / photo_scene | Traffic at 5-30 on Second Avenue, Detroit, Mich. LCCN2017878 | Arthur S. Siegel | 1942-07 |  |
| `commons_landscape_northeast_utah1a34285v` | photo / photo_scene | Landscape, Northeast Utah1a34285v | John Vachon | April 1942 | dev |
| `sdxl_00` | sdxl / sdxl_portrait | a studio portrait photograph of an elderly fisherman with a  | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_01` | sdxl / sdxl_portrait | a close-up photograph of a smiling young woman with freckles | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_02` | sdxl / sdxl_portrait | a portrait photograph of a child in a yellow raincoat lookin | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_03` | sdxl / sdxl_portrait | a photograph of a middle-aged man wearing glasses and a gree | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_04` | sdxl / sdxl_portrait | a candid photograph of two musicians laughing together on a  | Stable Diffusion XL base 1.0 (own genera |  | dev |
| `sdxl_05` | sdxl / sdxl_portrait | an oil painting of a young woman in a blue dress seated by a | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_06` | sdxl / sdxl_portrait | an impressionist oil painting of an old sailor smoking a pip | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_07` | sdxl / sdxl_portrait | a baroque oil painting portrait of a nobleman with a lace co | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_08` | sdxl / sdxl_portrait | a watercolor and gouache portrait of a girl with a red ribbo | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_09` | sdxl / sdxl_portrait | an expressionist oil painting of a laughing woman with bold  | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_10` | sdxl / sdxl_scene | a photograph of a mountain lake at sunrise with mist and pin | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_11` | sdxl / sdxl_scene | a photograph of a narrow cobblestone street in an old Europe | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_12` | sdxl / sdxl_scene | a photograph of a lighthouse on a rocky coast under a stormy | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_13` | sdxl / sdxl_scene | a photograph of a wheat field with a lone tree at golden hou | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_14` | sdxl / sdxl_scene | a photograph of a busy harbor with colorful fishing boats in | Stable Diffusion XL base 1.0 (own genera |  | dev |
| `sdxl_15` | sdxl / sdxl_scene | an impressionist oil painting of a river with water lilies a | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_16` | sdxl / sdxl_scene | an oil painting of a snow covered village at dusk with lit w | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_17` | sdxl / sdxl_scene | a romantic landscape painting of a ship in a stormy sea with | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_18` | sdxl / sdxl_scene | a post-impressionist painting of a sunflower field under a s | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_19` | sdxl / sdxl_scene | a Japanese woodblock print style image of large waves and a  | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_20` | sdxl / sdxl_object | a photograph of a red fox sitting in the snow looking at the | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_21` | sdxl / sdxl_object | a photograph of a bowl of ripe peaches on a wooden table, so | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_22` | sdxl / sdxl_object | a photograph of a vintage bicycle leaning against a blue wal | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_23` | sdxl / sdxl_object | a photograph of a ceramic teapot and two cups on a linen clo | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_24` | sdxl / sdxl_object | a photograph of a horse standing in a misty meadow | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_25` | sdxl / sdxl_object | a still life oil painting of a vase of flowers, lemons and a | Stable Diffusion XL base 1.0 (own genera |  | dev |
| `sdxl_26` | sdxl / sdxl_object | an oil painting of a white cat sleeping on a velvet cushion | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_27` | sdxl / sdxl_object | a gouache painting of a rooster with colorful feathers | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_28` | sdxl / sdxl_object | an oil painting of a bowl of cherries and a silver spoon on  | Stable Diffusion XL base 1.0 (own genera |  |  |
| `sdxl_29` | sdxl / sdxl_object | a charcoal and oil sketch of a horse running in a field | Stable Diffusion XL base 1.0 (own genera |  |  |

<!-- END EVALSET -->
