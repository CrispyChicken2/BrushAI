# Visual audit of the evaluation set (T-034)

Every one of the 100 prepared 512 x 512 images was looked at on contact sheets (4 x 3 images of 320 px per sheet,
`python -m scripts.build_eval_set sheets`) against the rules a script cannot check: a clear subject that matches
the stratum, no frame or foreign element over 5 % of the area, no text overlay, no possibly living identifiable person
(children included), no sepia tint where colour is claimed. The border column is the machine figure of `audit_eval_set`
(definition of the trimming, on the cropped source). A verdict `ok` means the image passes every rule; images that did not
pass were rejected during the audit, with a reason, in `configs/eval/curation_rejections.yaml`, and replaced by the next candidate
of the seeded draw. The final list below is the one that was audited after the last redraw.

| # | id | stratum | verdict | border | note |
|---|---|---|---|---|---|
| 1 | `met_695496` | european_portrait | ok | 0.0% | Bust of a Man (Saint Matthias?): clear subject for the stratum, no frame, mount or text overlay |
| 2 | `met_435862` | european_other | ok | 0.3% | The Route Nationale at Samer: clear subject for the stratum, no frame, mount or text overlay |
| 3 | `met_748947` | european_portrait | ok | 0.0% | Ellen Maurice (1578–1626): clear subject for the stratum, no frame, mount or text overlay |
| 4 | `met_436642` | european_other | ok | 0.3% | The Musician: clear subject for the stratum, no frame, mount or text overlay |
| 5 | `met_436262` (dev) | european_other | ok | 0.1% | Virgin and Child with Saint Catherine of Alexandria: clear subject for the stratum, no frame, mount or text overlay |
| 6 | `met_439977` | european_other | ok | 0.0% | Christ Carrying the Cross: clear subject for the stratum, no frame, mount or text overlay |
| 7 | `met_436496` | european_other | ok | 0.2% | thin dark strip at the bottom edge (about 2 %), part of the panel's gilt rim |
| 8 | `met_437921` | european_portrait | ok | 0.0% | Portrait of a Man: clear subject for the stratum, no frame, mount or text overlay |
| 9 | `met_438545` (dev) | european_other | ok | 0.6% | Léon Pallière (1787–1820) in His Room at the Villa Medi: clear subject for the stratum, no frame, mount or text overlay |
| 10 | `met_436616` | european_other | ok | 0.0% | Young Man and Woman in an Inn: clear subject for the stratum, no frame, mount or text overlay |
| 11 | `met_437897` | european_other | ok | 0.1% | Abraham's Parting from the Family of Lot: clear subject for the stratum, no frame, mount or text overlay |
| 12 | `met_436824` | european_portrait | ok | 0.1% | Lady Mary Berkeley: clear subject for the stratum, no frame, mount or text overlay |
| 13 | `met_437990` | european_portrait | ok | 0.1% | thin gilt frame strips at the left and right edges (about 2-3 % together) |
| 14 | `met_435993` | european_portrait | ok | 0.2% | Marianne Dorothy Harland (1759–1785), Later Mrs. Willia: clear subject for the stratum, no frame, mount or text overlay |
| 15 | `met_436253` | european_portrait | ok | 0.2% | Lucas van Uffel (died 1637): clear subject for the stratum, no frame, mount or text overlay |
| 16 | `met_437386` | european_portrait | ok | 0.1% | Portrait of a Man, probably a Member of the Van Bereste: clear subject for the stratum, no frame, mount or text overlay |
| 17 | `met_437211` | european_landscape | ok | 0.0% | The Rest on the Flight into Egypt: clear subject for the stratum, no frame, mount or text overlay |
| 18 | `met_436025` | european_landscape | ok | 0.1% | View of Ornans: clear subject for the stratum, no frame, mount or text overlay |
| 19 | `met_437179` | european_landscape | ok | 0.1% | Near Penshurst, Kent: clear subject for the stratum, no frame, mount or text overlay |
| 20 | `met_436782` | european_landscape | ok | 0.1% | Capriccio with St. Paul's and Old London Bridge: clear subject for the stratum, no frame, mount or text overlay |
| 21 | `met_436183` | european_landscape | ok | 0.1% | Departure of the Amazons: clear subject for the stratum, no frame, mount or text overlay |
| 22 | `met_436830` | european_landscape | ok | 0.1% | A Panoramic Landscape with a Country Estate: clear subject for the stratum, no frame, mount or text overlay |
| 23 | `met_436971` | european_landscape | ok | 0.5% | The Flood Gate: clear subject for the stratum, no frame, mount or text overlay |
| 24 | `met_435880` | european_landscape | ok | 0.0% | Rocks at Fontainebleau: clear subject for the stratum, no frame, mount or text overlay |
| 25 | `met_72416` | asian | ok | 0.0% | thin dark mount strip along the top edge (about 1-2 %) |
| 26 | `met_54318` | asian | ok | 0.0% | calligraphy block is part of the print (integral inscription, under a third of the area) |
| 27 | `met_54196` (dev) | asian | ok | 0.0% | inscriptions and seals are part of the print (under a third of the area) |
| 28 | `met_54675` | asian | ok | 0.0% | Bird Catching Fish Among Reeds: clear subject for the stratum, no frame, mount or text overlay |
| 29 | `met_48980` | asian | ok | 0.0% | thin dark strip along the top edge (about 2 %) |
| 30 | `met_54224` | asian | ok | 0.0% | inscriptions and a cartouche are part of the print (under a third of the area) |
| 31 | `met_54923` | asian | ok | 0.0% | inscriptions are part of the print (under a third of the area) |
| 32 | `met_74670` | asian | ok | 0.0% | ink drawing of a feline on paper, clear subject |
| 33 | `met_488147` | other_art | ok | 0.0% | Still Life: clear subject for the stratum, no frame, mount or text overlay |
| 34 | `met_488126` | other_art | ok | 0.1% | "One Third of a Nation": clear subject for the stratum, no frame, mount or text overlay |
| 35 | `met_485551` | other_art | ok | 0.0% | Self-Portrait: clear subject for the stratum, no frame, mount or text overlay |
| 36 | `met_488903` | other_art | ok | 0.0% | Jeanne Hébuterne: clear subject for the stratum, no frame, mount or text overlay |
| 37 | `met_11718` | other_art | ok | 0.0% | Samuel Mifflin: clear subject for the stratum, no frame, mount or text overlay |
| 38 | `met_701989` | other_art | ok | 0.0% | thin dark frame strip at the left edge (about 3 %); a child, but a portrait painted in the 19th century |
| 39 | `met_11241` | other_art | ok | 0.0% | Augustus Washington Clason: clear subject for the stratum, no frame, mount or text overlay |
| 40 | `met_13102` (dev) | other_art | ok | 0.0% | grey-scale portrait painting, clear subject |
| 41 | `met_283187` (dev) | photo_portrait | ok | 0.0% | couple, 19th-century carte, sitters long dead |
| 42 | `met_285903` | photo_portrait | ok | 0.0% | a girl, 19th-century sitter, long dead (the rule on children concerns people who can be alive) |
| 43 | `met_261532` | photo_portrait | ok | 0.0% | Marie Stuart: clear subject for the stratum, no frame or text overlay, no identifiable living person |
| 44 | `met_263532` | photo_portrait | ok | 0.0% | hand-coloured albumen print, counted in colour; sitter long dead |
| 45 | `met_268534` | photo_portrait | ok | 0.0% | group portrait of four sitters, 19th century |
| 46 | `met_261324` | photo_portrait | ok | 0.0% | Derelitta: clear subject for the stratum, no frame or text overlay, no identifiable living person |
| 47 | `met_268532` | photo_portrait | ok | 0.0% | Dr. Welsh: clear subject for the stratum, no frame or text overlay, no identifiable living person |
| 48 | `met_285608` | photo_portrait | ok | 0.0% | sitter holds a lens-mask over her face (Pierson, 19th century); long dead |
| 49 | `met_286569` | photo_portrait | ok | 0.0% | sepia portrait, pale mount strip at the left edge (about 3 %) |
| 50 | `commons_guide_at_little_norway_blue_mounds_wis_lccn2017878934` | photo_portrait | ok | 0.0% | adult sitter in a 1940-43 slide, black film edge trimmed; an adult of that date cannot be alive |
| 51 | `commons_fox_squirrell_51384470246` | photo_animal | ok | 0.0% | agency photograph, squirrel on a trunk, no person (detector: none) |
| 52 | `commons_dehydrated_key_deer_36952540200` | photo_animal | ok | 0.0% | agency photograph, a rescued deer drinking from a tub, no person visible (detector: none) |
| 53 | `commons_trumpeter_swan_at_seedskadee_national_wildlife_refuge_497247` | photo_animal | ok | 0.0% | agency photograph, swan on the water, no person |
| 54 | `commons_short_eared_owl_at_funk_waterfowl_production_area_4100881413` | photo_animal | ok | 0.0% | agency photograph, owl on a sign post, the sign is cut at the bottom edge (under 5 %) |
| 55 | `commons_ruddy_duck_scott_somershoe_49717209451` (dev) | photo_animal | ok | 0.0% | agency photograph, duck on the water, no person |
| 56 | `commons_white_tailed_deer_53627568015` | photo_animal | ok | 0.0% | agency photograph, two deer, no person |
| 57 | `commons_wild_turkey_52870186395` | photo_animal | ok | 0.0% | agency photograph, turkey in display, no person |
| 58 | `commons_harrowing_a_field_with_a_diesel_tractor_seabrook_farm_bridge` | photo_object | ok | 0.0% | 1942 slide, tractor and harrow small but the subject; no person visible; dark film edge trimmed |
| 59 | `commons_street_in_san_juan_1a34067v` | photo_object | ok | 0.1% | 1941 slide; two or three distant figures, too small to identify; thin dark bar at the bottom (about 3 %) |
| 60 | `commons_door_and_window_in_a_spanish_american_home_1a34162v` | photo_object | ok | 2.2% | 1940 slide, door and window of an adobe house; black bar at the right edge (about 3 %) |
| 61 | `commons_laundry_barbershop_and_stores_washington_d_c_1a34426v` | photo_object | ok | 0.0% | 1942 slide; one small walking figure, not identifiable; sign text is part of the shopfront |
| 62 | `commons_russellllanodesanjuan` | photo_object | ok | 0.0% | 1940s slide, adobe church, no person; clear subject (object stratum: a building) |
| 63 | `commons_seed_and_feed_store_lincoln_nebraska1a34278v` | photo_object | ok | 0.1% | 1942 slide; adults seen from behind, not identifiable; the dark band at the top is the shadow of an awning (content) |
| 64 | `commons_near_white_plains_ga_lccn2017877524` | photo_scene | ok | 4.5% | 1941 slide; an adult woman small in the doorway, not identifiable (an adult of 1941 cannot be alive) |
| 65 | `commons_a_street_in_a_town_of_the_virgin_islands1a33999v` | photo_scene | ok | 2.0% | 1941 slide; cars small in the street; dark edges of the film below 5 % |
| 66 | `commons_scrap_and_salvage_depot_butte_mont_lccn2017878813` | photo_scene | ok | 0.1% | 1942 slide, scrap heap; no person; dark edge below 5 % |
| 67 | `commons_russell_lee_street_corner_dillon_montana` | photo_scene | ok | 0.0% | 1942 slide, a brick corner building on a street, a few tiny figures, not identifiable; no border |
| 68 | `commons_the_greensboro_lumber_co_greensboro_ga_lccn2017877531` | photo_scene | ok | 0.1% | 1941 slide, lumber yard under a stormy sky, no person |
| 69 | `commons_traffic_at_5_30_on_second_avenue_detroit_mich_lccn2017878611` | photo_scene | ok | 0.1% | 1942 aerial view of a street; vehicles small, no person identifiable |
| 70 | `commons_landscape_northeast_utah1a34285v` (dev) | photo_scene | ok | 0.3% | 1941 slide, landscape with a dead tree trunk, thin dark edges (below 5 %) |
| 71 | `sdxl_00` | sdxl_portrait | ok | - | a studio portrait photograph of an elderly fisherman wi: synthetic image, clear subject, no text, no frame; no real person |
| 72 | `sdxl_01` | sdxl_portrait | ok | - | a close-up photograph of a smiling young woman with fre: synthetic image, clear subject, no text, no frame; no real person |
| 73 | `sdxl_02` | sdxl_portrait | ok | - | a portrait photograph of a child in a yellow raincoat l: synthetic image, clear subject, no text, no frame; no real person |
| 74 | `sdxl_03` | sdxl_portrait | ok | - | a photograph of a middle-aged man wearing glasses and a: synthetic image, clear subject, no text, no frame; no real person |
| 75 | `sdxl_04` (dev) | sdxl_portrait | ok | - | a candid photograph of two musicians laughing together : synthetic image, clear subject, no text, no frame; no real person |
| 76 | `sdxl_05` | sdxl_portrait | ok | - | an oil painting of a young woman in a blue dress seated: synthetic image, clear subject, no text, no frame; no real person |
| 77 | `sdxl_06` | sdxl_portrait | ok | - | an impressionist oil painting of an old sailor smoking : synthetic image, clear subject, no text, no frame; no real person |
| 78 | `sdxl_07` | sdxl_portrait | ok | - | a baroque oil painting portrait of a nobleman with a la: synthetic image, clear subject, no text, no frame; no real person |
| 79 | `sdxl_08` | sdxl_portrait | ok | - | a watercolor and gouache portrait of a girl with a red : synthetic image, clear subject, no text, no frame; no real person |
| 80 | `sdxl_09` | sdxl_portrait | ok | - | an expressionist oil painting of a laughing woman with : synthetic image, clear subject, no text, no frame; no real person |
| 81 | `sdxl_10` | sdxl_scene | ok | - | a photograph of a mountain lake at sunrise with mist an: synthetic image, clear subject, no text, no frame; no real person |
| 82 | `sdxl_11` | sdxl_scene | ok | - | a photograph of a narrow cobblestone street in an old E: synthetic image, clear subject, no text, no frame; no real person |
| 83 | `sdxl_12` | sdxl_scene | ok | - | a photograph of a lighthouse on a rocky coast under a s: synthetic image, clear subject, no text, no frame; no real person |
| 84 | `sdxl_13` | sdxl_scene | ok | - | a photograph of a wheat field with a lone tree at golde: synthetic image, clear subject, no text, no frame; no real person |
| 85 | `sdxl_14` (dev) | sdxl_scene | ok | - | a photograph of a busy harbor with colorful fishing boa: synthetic image, clear subject, no text, no frame; no real person |
| 86 | `sdxl_15` | sdxl_scene | ok | - | an impressionist oil painting of a river with water lil: synthetic image, clear subject, no text, no frame; no real person |
| 87 | `sdxl_16` | sdxl_scene | ok | - | an oil painting of a snow covered village at dusk with : synthetic image, clear subject, no text, no frame; no real person |
| 88 | `sdxl_17` | sdxl_scene | ok | - | a romantic landscape painting of a ship in a stormy sea: synthetic image, clear subject, no text, no frame; no real person |
| 89 | `sdxl_18` | sdxl_scene | ok | - | synthetic; a swirling sky over sunflowers, a pastiche of the motif of a tuning image (data card, overlap section) |
| 90 | `sdxl_19` | sdxl_scene | ok | - | synthetic; waves and a distant mountain in woodblock style, a pastiche of a tuning image (data card); thin drawn border (1-2 %) |
| 91 | `sdxl_20` | sdxl_object | ok | - | a photograph of a red fox sitting in the snow looking a: synthetic image, clear subject, no text, no frame; no real person |
| 92 | `sdxl_21` | sdxl_object | ok | - | a photograph of a bowl of ripe peaches on a wooden tabl: synthetic image, clear subject, no text, no frame; no real person |
| 93 | `sdxl_22` | sdxl_object | ok | - | a photograph of a vintage bicycle leaning against a blu: synthetic image, clear subject, no text, no frame; no real person |
| 94 | `sdxl_23` | sdxl_object | ok | - | a photograph of a ceramic teapot and two cups on a line: synthetic image, clear subject, no text, no frame; no real person |
| 95 | `sdxl_24` | sdxl_object | ok | - | a photograph of a horse standing in a misty meadow: synthetic image, clear subject, no text, no frame; no real person |
| 96 | `sdxl_25` (dev) | sdxl_object | ok | - | a still life oil painting of a vase of flowers, lemons : synthetic image, clear subject, no text, no frame; no real person |
| 97 | `sdxl_26` | sdxl_object | ok | - | an oil painting of a white cat sleeping on a velvet cus: synthetic image, clear subject, no text, no frame; no real person |
| 98 | `sdxl_27` | sdxl_object | ok | - | a gouache painting of a rooster with colorful feathers: synthetic image, clear subject, no text, no frame; no real person |
| 99 | `sdxl_28` | sdxl_object | ok | - | an oil painting of a bowl of cherries and a silver spoo: synthetic image, clear subject, no text, no frame; no real person |
| 100 | `sdxl_29` | sdxl_object | ok | - | a charcoal and oil sketch of a horse running in a field: synthetic image, clear subject, no text, no frame; no real person |
