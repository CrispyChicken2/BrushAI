| strokes | path | chunk | checkpoint | time / step | peak VRAM | max image dev | max grad dev (rel) |
|---|---|---|---|---|---|---|---|
| 64 | full | 64 | no | 39 ms | 1.15 GB | 0 | 0 |
| 64 | bbox | 128 | no | 38 ms (1.0x) | 1.16 GB | 0.0e+00 | 0.0e+00 |
| 64 | bbox | 64 | no | 39 ms (1.0x) | 1.16 GB | 0.0e+00 | 0.0e+00 |
| 64 | bbox | 32 | no | 31 ms (1.3x) | 0.60 GB | 1.8e-07 | 1.3e-06 |
| 64 | bbox | 16 | no | 41 ms (1.0x) | 0.50 GB | 1.8e-07 | 5.8e-07 |
| 256 | full | 64 | no | 152 ms | 3.18 GB | 0 | 0 |
| 256 | bbox | 128 | no | 141 ms (1.1x) | 3.55 GB | 1.8e-07 | 2.2e-06 |
| 256 | bbox | 64 | no | 138 ms (1.1x) | 2.70 GB | 0.0e+00 | 1.7e-08 |
| 256 | bbox | 32 | no | 124 ms (1.2x) | 1.76 GB | 1.2e-07 | 1.9e-06 |
| 256 | bbox | 16 | no | 176 ms (0.9x) | 1.43 GB | 1.8e-07 | 1.6e-06 |
| 1024 | full | 64 | yes | 771 ms | 1.15 GB | 0 | 0 |
| 1024 | bbox | 128 | yes | 377 ms (2.0x) | 2.17 GB | 1.8e-07 | 4.2e-06 |
| 1024 | bbox | 64 | yes | 378 ms (2.0x) | 1.11 GB | 0.0e+00 | 6.5e-09 |
| 1024 | bbox | 32 | yes | 492 ms (1.6x) | 0.59 GB | 1.2e-07 | 4.2e-06 |
| 1024 | bbox | 16 | yes | 914 ms (0.8x) | 0.35 GB | 1.8e-07 | 4.2e-06 |
| 2037 | full | 64 | yes | 1514 ms | 1.19 GB | 0 | 0 |
| 2037 | bbox | 128 | yes | 640 ms (2.4x) | 2.20 GB | 1.8e-07 | 4.1e-06 |
| 2037 | bbox | 64 | yes | 641 ms (2.4x) | 1.16 GB | 1.2e-07 | 5.1e-08 |
| 2037 | bbox | 32 | yes | 967 ms (1.6x) | 0.69 GB | 1.2e-07 | 3.8e-06 |
| 2037 | bbox | 16 | yes | 1853 ms (0.8x) | 0.54 GB | 1.2e-07 | 2.7e-06 |
