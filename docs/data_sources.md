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
```

**Sources and licences (checked 2026-10-26).**
- *Paintings (40)*: Metropolitan Museum Open Access, `isPublicDomain = true`, CC0 (API v1.1, object list per department; the `search` endpoint
  was retired on 2026-10-01). 24 European paintings (8 portraits / 8 landscapes / 8 other, from the Met subject tags), 8 Asian (4 paintings,
  4 woodblock prints), 8 other (4 Modern Art, 4 American Wing).
- *Photographs (30)*: 8 nineteenth-century portrait photographs of the Met (CC0, monochrome or sepia prints, sitters long dead); 22 colour
  photographs from Wikimedia Commons whose file page carries a public-domain tag and whose source is a US federal collection: 18 Library of
  Congress FSA/OWI Kodachrome slides (1939-1943, US government work), 3 US Fish and Wildlife Service photographs, 1 Library of Congress
  Prokudin-Gorsky colour photograph (1900s, tag "Public domain"). 10 portraits (8 + 2), 7 animals, 6 objects, 7 scenes. Smithsonian Open
  Access was not needed and is not used, so its per-file statement was not relied on. Excluded: any image of a person who may be alive
  (children of the FSA series are excluded by the title rule and by the contact-sheet check).
- *SDXL (30)*: Stable Diffusion XL base 1.0, revision `462165984030d82259a11f4367a4eed129e94a7b`, licence CreativeML Open RAIL++-M (Licensor
  claims no rights in the outputs; the use restrictions do not concern the prompts). The 30 outputs are dedicated to CC0 1.0. Settings are
  the library defaults (`configs/eval/sdxl_prompts.yaml`, dependencies pinned in `requirements-gen.txt`); fp16 weights with model CPU offload; `diffusers` 0.42.0, `torch` 2.11.0+cu128.
  Measured on the RTX 5070: peak VRAM 5.3 GB allocated / 7.5 GB reserved, 20 to 32 s per image (20 s alone, 32 s with the CPU busy).

**Selection (all mechanical, none looks at a method result).** `random.Random(22)`; per department or category the candidate ids sorted then
shuffled; the first candidates that pass the rules are kept (`scripts/build_eval_set.py`, docstring). Rules: public domain / CC0 with a
per-file statement, source short side >= 768 px and aspect in [3:4, 4:3], colour for the Commons photographs, at most 3 images per
photographer, no near-duplicate of an accepted or of a tuning image. The 21 visual rejections taken on the contact sheets (text over 5 %
of the area, frames and mats, no clear subject, possible living person, sepia tint, duplicates) are listed with their reason in
`configs/eval/curation_rejections.yaml`, and every rejected candidate was replaced by the next one of the seeded draw. Preparation: centre crop
(43 images carry a mechanical `crop_box` that removes uniform black film or mat borders, side >= 640 px) and a Pillow BOX (area) reduction to 512.

**Reproducibility, measured.**
- Fetch: a download from scratch into an empty directory reproduced the prepared pixels of all 70 images (SHA-256) in 57 s. Two Met files had
  changed *bytes* between the curation and the check (`met_488147`, `met_261318`: the museum re-encoded them, same decoded pixels), so the
  verification decides on the prepared pixels and reports the file difference.
- SDXL: two complete generations on the same machine are bitwise identical (30 of 30 images, max difference 0), and a third run verified all
  the hashes of the manifest. GPU kernels are not guaranteed bitwise stable across drivers and devices: `generate_eval_sdxl` reports the pixel
  difference of a mismatch (`--allow-mismatch` to continue); the hashes in the manifest are those of this machine (record: `docs/experiments/data/eval_set/sdxl_generation.json`).
- Overlap with tuning: the six images of the T-009 sweep and the M1/M2 demos (Mona Lisa, Starry Night, Great Wave, Pearl Earring, Earth from
  Apollo 17, Kirkjufell) are not in the set; the largest 32 x 32 thumbnail correlation between any set image and any of them is 0.59
  (threshold 0.9); the Met impression of the Great Wave (object 45434) was excluded from the draw by id. Record: `docs/experiments/data/eval_set/overlap_check.json`.

**Dev split (10, fixed here before any method ran).** 4 paintings (2 European, 1 Asian, 1 other), 3 photographs (1 portrait, 2 others), SDXL
prompts 04, 14, 25. Drawn with the same `random.Random(22)` after the three draws, flagged `dev: true` in the manifest and in the table below.

**Limits.** The photographs are dominated by one US federal collection (FSA/OWI 1939-1943) and the portraits by 19th-century prints, so they
are not representative of modern photography; the paintings are mostly European; some scanned slides keep a thin dark border under 5 % of the area; inscriptions that are part of an East Asian work are kept (below a third of the area).

<!-- BEGIN EVALSET -->

| category | stratum | n | dev | licence |
|---|---|---|---|---|
| painting | asian | 8 | 1 | CC0 1.0 (Met Open Access, isPublicDomain) |
| painting | european_landscape | 8 | 0 | CC0 1.0 (Met Open Access, isPublicDomain) |
| painting | european_other | 8 | 1 | CC0 1.0 (Met Open Access, isPublicDomain) |
| painting | european_portrait | 8 | 1 | CC0 1.0 (Met Open Access, isPublicDomain) |
| painting | other_art | 8 | 1 | CC0 1.0 (Met Open Access, isPublicDomain) |
| photo | photo_animal | 5 | 0 | Public domain (US government work, Library of Congress FSA/OWI) |
| photo | photo_animal | 1 | 0 | Public domain (US government work, US Fish and Wildlife Service) |
| photo | photo_animal | 1 | 0 | Public domain (Wikimedia Commons licence tag) |
| photo | photo_object | 6 | 1 | Public domain (US government work, Library of Congress FSA/OWI) |
| photo | photo_portrait | 8 | 1 | CC0 1.0 (Met Open Access, isPublicDomain) |
| photo | photo_portrait | 2 | 1 | Public domain (US government work, Library of Congress FSA/OWI) |
| photo | photo_scene | 5 | 1 | Public domain (US government work, Library of Congress FSA/OWI) |
| photo | photo_scene | 2 | 1 | Public domain (US government work, US Fish and Wildlife Service) |
| sdxl | sdxl_object | 10 | 1 | CC0 1.0 (own-generated, SDXL output) |
| sdxl | sdxl_portrait | 10 | 1 | CC0 1.0 (own-generated, SDXL output) |
| sdxl | sdxl_scene | 10 | 1 | CC0 1.0 (own-generated, SDXL output) |

| id | category / stratum | title | author | date | dev |
|---|---|---|---|---|---|
| `met_695496` | painting / european_portrait | Bust of a Man (Saint Matthias?) | Giovanni Battista Piazzetta | ca. 1715–20 | dev |
| `met_435862` | painting / european_other | The Route Nationale at Samer | Jean-Charles Cazin |  |  |
| `met_748947` | painting / european_portrait | Ellen Maurice (1578–1626) | Marcus Gheeraerts the Younger | 1597 |  |
| `met_436642` | painting / european_other | The Musician | Bartholomeus van der Helst | 1662 |  |
| `met_436262` | painting / european_other | Virgin and Child with Saint Catherine of Alexandria | Anthony van Dyck | ca. 1630 |  |
| `met_439977` | painting / european_other | Christ Carrying the Cross | Jan Gossart (called Mabuse) | ca. 1520–25 |  |
| `met_436496` | painting / european_other | Saint John the Evangelist with Acteus and Eugenius | Francescuccio Ghissi (Francesco di Cecco | ca. 1370 |  |
| `met_437921` | painting / european_portrait | Portrait of a Man | Abraham de Vries | 1643 |  |
| `met_438545` | painting / european_other | Léon Pallière (1787–1820) in His Room at the Villa Medici, R | Jean Alaux | 1817 |  |
| `met_436616` | painting / european_other | Young Man and Woman in an Inn | Frans Hals | 1623 | dev |
| `met_437897` | painting / european_other | Abraham's Parting from the Family of Lot | Jan Victors | ca. 1655–65 |  |
| `met_436824` | painting / european_portrait | Lady Mary Berkeley | Sir Godfrey Kneller | ca. 1700 |  |
| `met_437548` | painting / european_landscape | Mountain Torrent | Jacob van Ruisdael | 1670s |  |
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
| `met_72416` | painting / asian | Landscape | Watanabe Shikō 渡辺始興 | first half of the 18th century |  |
| `met_54318` | painting / asian | Kanjo (Court Lady) Seated, and a Tsurikoro Hanging near Her  | Totoya Hokkei | ca. 1825 |  |
| `met_54196` | painting / asian | Snowy Landscape | Ishikawa Kazan | ca. 1820 |  |
| `met_54675` | painting / asian | Bird Catching Fish Among Reeds | Watanabe Seitei | ca. 1887 |  |
| `met_48980` | painting / asian | Spring and Summer Flowers with Bamboo | unknown | 18th century |  |
| `met_54224` | painting / asian | Print | Totoya Hokkei | 1780–1850 |  |
| `met_54923` | painting / asian | Nanamaro and His Followers Looking at the Moon in China | Kubo Shunman | 19th century | dev |
| `met_51686` | painting / asian | Tartar Officer Carrying Blond Lady | Unidentified artist | 19th century |  |
| `met_488147` | painting / other_art | Still Life | Ben Benn | 1941 |  |
| `met_488126` | painting / other_art | "One Third of a Nation" | O. Louis Guglielmi | 1939 |  |
| `met_485551` | painting / other_art | Self-Portrait | Umberto Boccioni | 1905 | dev |
| `met_488903` | painting / other_art | Jeanne Hébuterne | Amedeo Modigliani | 1919 |  |
| `met_11718` | painting / other_art | Samuel Mifflin | Charles Willson Peale | 1777–80 |  |
| `met_701989` | painting / other_art | Emma Van Name | Joshua Johnson | ca. 1805 |  |
| `met_11241` | painting / other_art | Augustus Washington Clason | John Wesley Jarvis | ca. 1820 |  |
| `met_13102` | painting / other_art | Portrait of a Lady (Possibly of the Buloid Family) | Samuel Lovett Waldo | 1832 |  |
| `met_283187` | photo / photo_portrait | [Senator and Mrs. James Henry Lane] | Brady & Co. | 1861–66 |  |
| `met_285903` | photo / photo_portrait | Portrait of His Daughter Reading | Louis-Désiré Blanquart-Évrard | ca. 1847 |  |
| `met_261532` | photo / photo_portrait | Marie Stuart | Pierre-Louis Pierson | 1860s |  |
| `met_263532` | photo / photo_portrait | [Performer in Bugaku-style Costume] | Unknown | 1880s |  |
| `met_268534` | photo / photo_portrait | Cunningham, Beff, John Hamilton, Guthrie | Hill and Adamson | 1843–47 |  |
| `met_261324` | photo / photo_portrait | Derelitta | Pierre-Louis Pierson | 1860s | dev |
| `met_261318` | photo / photo_portrait | Sèriè à la Ristori | Pierre-Louis Pierson | 1860s |  |
| `met_268532` | photo / photo_portrait | Dr. Welsh | Hill and Adamson | 1843–47 |  |
| `commons_gus_worke_a_farmer_who_came_from_germany_40_years_ago1a35004` | photo / photo_portrait | Gus Worke, a farmer who came from Germany 40 years ago1a3500 | Charles Fenno Jacobs | May 1942date QS:P571,+1942-05-00T00:00:00Z/10 |  |
| `commons_guide_at_little_norway_blue_mounds_wis_lccn2017878934` | photo / photo_portrait | Guide at Little Norway, Blue Mounds, Wis. LCCN2017878934 | Arthur Rothstein | 1942-08 |  |
| `commons_gorsky21767_3` | photo / photo_animal | Gorsky21767-3 | Sergei Mikhailovich Prokudin-Gorskii, di | 1911-01-01 |  |
| `commons_south_andros_island_piping_plover_foraging_in_rocky_intertid` | photo / photo_animal | South Andros Island- Piping Plover foraging in rocky interti | U. S. Fish and Wildlife Service - Northe | 2011-02-01 15:55:22 |  |
| `commons_cattle_in_corrals_on_ranch_beaverhead_county1a35024v` | photo / photo_animal | Cattle in corrals on ranch, Beaverhead County1a35024v | Russell Lee | September 1942date QS:P571,+1942-09-00T00:00:00Z/10 |  |
| `commons_street_corner_dillon_mont1a35017v` | photo / photo_animal | Street corner, Dillon, Mont1a35017v | Russell Lee | August 1942date QS:P571,+1942-08-00T00:00:00Z/10 |  |
| `commons_horse_breeding_ranch_grant_co_oregon_lccn2017877601` | photo / photo_animal | Horse breeding ranch, Grant Co., Oregon LCCN2017877601 | Lee, Russell, 1903-1986, photographer | 1942-07 |  |
| `commons_russellshepherd` | photo / photo_animal | Russellshepherd | Russell Lee | 1942-08 |  |
| `commons_in_the_flat_cattle_country_in_the_south_western_tip_1a34078v` | photo / photo_animal | In the flat cattle country in the south western tip 1a34078v | Jack Delano | January 1942date QS:P571,+1942-01-00T00:00:00Z/10 |  |
| `commons_shulman_s_market_at_the_southeast_corner_of_n_street_and_uni` | photo / photo_object | Shulman's Market at the southeast corner of N Street and Uni | Louise Rosskam | between 1941 and 1942date QS:P,+1941-00-00T00:00:00Z/8,P1319,+1941-00-00T00:00:00Z/9,P1326,+1942-00-00T00:00:00Z/9 |  |
| `commons_seed_and_feed_store_lincoln_nebraska1a34278v` | photo / photo_object | Seed and feed store, Lincoln Nebraska | John Vachon | 1942date QS:P571,+1942-00-00T00:00:00Z/9 |  |
| `commons_street_in_san_juan_1a34067v` | photo / photo_object | Street in San Juan 1a34067v | Jack Delano | 1941-12 |  |
| `commons_rosskam_dc1` | photo / photo_object | Rosskam-dc1 | Louise Rosskam | between 1941 and 1942date QS:P,+1941-00-00T00:00:00Z/8,P1319,+1941-00-00T00:00:00Z/9,P1326,+1942-00-00T00:00:00Z/9 | dev |
| `commons_laundry_barbershop_and_store_washington_d_c_1a34422v` | photo / photo_object | [Laundry, barbershop and store, Washington, D.C.?] | Louise Rosskam | between 1941 and 1942date QS:P571,+1941-00-00T00:00:00Z/8,P1319,+1941-00-00T00:00:00Z/9,P1326,+1942-00-00T00:00:00Z/9 |  |
| `commons_harrowing_a_field_with_a_diesel_tractor_seabrook_farm_bridge` | photo / photo_object | Harrowing a field with a diesel tractor, Seabrook Farm, Brid | Collier, John, Jr., 1913-1992, photograp | 1942-06 |  |
| `commons_french_village_a_small_settlement_on_st_thomas_island1a33940` | photo / photo_scene | French village, a small settlement on St. Thomas Island1a339 | Jack Delano | December 1941date QS:P571,+1941-12-00T00:00:00Z/10 |  |
| `commons_fish_survey_at_air_force_base_51253092996` | photo / photo_scene | Fish survey at Air Force Base (51253092996) | USFWS Mountain-Prairie | Taken on 4 September 2014, 09:36 |  |
| `commons_part_of_string_bean_factory_and_field_system_seabrook_farm_b` | photo / photo_scene | Part of string bean factory and field system, Seabrook Farm, | Collier, John, Jr., 1913-1992, photograp | 1942-06 | dev |
| `commons_near_white_plains_ga_ca_1941_lccn2017877525` | photo / photo_scene | Near White Plains, Ga., ca. 1941. LCCN2017877525 | Delano, Jack, photographer; Wolcott, Mar | 1941 |  |
| `commons_shasta_dam_under_construction_california_lccn2017877975` | photo / photo_scene | Shasta dam under construction, California LCCN2017877975 | Lee, Russell, 1903-1986, photographer | 1942-06 |  |
| `commons_patuxent_research_refuge_md_10408633954` | photo / photo_scene | Patuxent Research Refuge, MD (10408633954) | U.S. Fish and Wildlife Service Headquart | 2002-10-26 09:38 |  |
| `commons_house_near_white_plains_ga_lccn2017877526` | photo / photo_scene | House near White Plains, Ga.? LCCN2017877526 | Delano, Jack, photographer; Wolcott, Mar | 1941 |  |
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
