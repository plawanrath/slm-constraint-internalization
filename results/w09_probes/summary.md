# Scope probes and activation patching

## SmolLM2-360M-Instruct:base

programs: 200 (gold 200, gen 0); sequences: 1051; layers: 32

| design | n | pos rate | best layer | acc [95% CI] | control | first layer | last layer |
|---|---|---|---|---|---|---|---|
| use | 1702 | 0.50 | 27 | 0.947 [0.937, 0.957] | 0.484 | 0.500 | 0.941 |
| def_diff | 774 | 0.50 | 11 | 0.988 [0.981, 0.995] | 0.492 | 0.956 | 0.953 |
| name | 1702 | 0.50 | 31 | 0.996 [0.994, 0.999] | 0.491 | 0.501 | 0.996 |

patching: 200 pairs (180 with a clean-corrupt gap > 0.1 nat); target log-prob clean -0.97 vs corrupt -5.55; peak target recovery at layer 30: 1.00 [1.00, 1.01], margin delta nan

| layer | target recovery | margin delta |
|---|---|---|
| 0 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 1 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 2 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 3 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 4 | -0.00 [-0.01, 0.00] | nan [nan, nan] |
| 5 | 0.00 [-0.01, 0.01] | nan [nan, nan] |
| 6 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 7 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 8 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 9 | 0.00 [-0.01, 0.01] | nan [nan, nan] |
| 10 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 11 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 12 | 0.01 [0.00, 0.01] | nan [nan, nan] |
| 13 | 0.01 [0.00, 0.01] | nan [nan, nan] |
| 14 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 15 | 0.00 [-0.01, 0.01] | nan [nan, nan] |
| 16 | 0.00 [-0.01, 0.01] | nan [nan, nan] |
| 17 | 0.01 [-0.00, 0.02] | nan [nan, nan] |
| 18 | 0.02 [0.00, 0.04] | nan [nan, nan] |
| 19 | 0.03 [0.01, 0.05] | nan [nan, nan] |
| 20 | -0.01 [-0.08, 0.04] | nan [nan, nan] |
| 21 | 0.02 [-0.05, 0.07] | nan [nan, nan] |
| 22 | 0.05 [-0.03, 0.11] | nan [nan, nan] |
| 23 | 0.13 [0.04, 0.19] | nan [nan, nan] |
| 24 | 0.68 [0.62, 0.74] | nan [nan, nan] |
| 25 | 0.66 [0.59, 0.72] | nan [nan, nan] |
| 26 | 0.80 [0.68, 0.89] | nan [nan, nan] |
| 27 | 0.94 [0.90, 0.97] | nan [nan, nan] |
| 28 | 0.95 [0.91, 0.97] | nan [nan, nan] |
| 29 | 0.97 [0.94, 0.99] | nan [nan, nan] |
| 30 | 1.00 [1.00, 1.01] | nan [nan, nan] |
| 31 | 1.00 [1.00, 1.00] | nan [nan, nan] |

## SmolLM2-360M-Instruct:smollm2-360m-instruct_ssd_s0

programs: 310 (gold 200, gen 110); sequences: 1454; layers: 32

| design | n | pos rate | best layer | acc [95% CI] | control | first layer | last layer |
|---|---|---|---|---|---|---|---|
| use | 2310 | 0.50 | 27 | 0.948 [0.939, 0.958] | 0.509 | 0.498 | 0.945 |
| def_diff | 870 | 0.50 | 19 | 0.986 [0.978, 0.993] | 0.509 | 0.956 | 0.954 |
| name | 2310 | 0.50 | 31 | 0.992 [0.988, 0.996] | 0.491 | 0.503 | 0.992 |

patching: 200 pairs (185 with a clean-corrupt gap > 0.1 nat); target log-prob clean -0.82 vs corrupt -6.90; peak target recovery at layer 30: 1.00 [1.00, 1.00], margin delta nan

| layer | target recovery | margin delta |
|---|---|---|
| 0 | 0.00 [-0.00, 0.00] | nan [nan, nan] |
| 1 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 2 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 3 | -0.00 [-0.01, 0.00] | nan [nan, nan] |
| 4 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 5 | 0.00 [0.00, 0.01] | nan [nan, nan] |
| 6 | 0.00 [-0.01, 0.01] | nan [nan, nan] |
| 7 | 0.00 [-0.01, 0.01] | nan [nan, nan] |
| 8 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 9 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 10 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 11 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 12 | 0.01 [0.00, 0.01] | nan [nan, nan] |
| 13 | 0.01 [-0.00, 0.01] | nan [nan, nan] |
| 14 | 0.00 [0.00, 0.01] | nan [nan, nan] |
| 15 | 0.00 [-0.01, 0.01] | nan [nan, nan] |
| 16 | -0.00 [-0.01, 0.01] | nan [nan, nan] |
| 17 | 0.00 [-0.00, 0.01] | nan [nan, nan] |
| 18 | 0.01 [-0.00, 0.02] | nan [nan, nan] |
| 19 | 0.02 [0.00, 0.03] | nan [nan, nan] |
| 20 | 0.02 [-0.02, 0.05] | nan [nan, nan] |
| 21 | 0.06 [-0.00, 0.10] | nan [nan, nan] |
| 22 | 0.09 [0.02, 0.15] | nan [nan, nan] |
| 23 | 0.20 [0.13, 0.26] | nan [nan, nan] |
| 24 | 0.75 [0.69, 0.81] | nan [nan, nan] |
| 25 | 0.70 [0.57, 0.79] | nan [nan, nan] |
| 26 | 0.82 [0.64, 0.92] | nan [nan, nan] |
| 27 | 0.91 [0.77, 0.99] | nan [nan, nan] |
| 28 | 0.92 [0.82, 0.98] | nan [nan, nan] |
| 29 | 0.95 [0.88, 0.99] | nan [nan, nan] |
| 30 | 1.00 [1.00, 1.00] | nan [nan, nan] |
| 31 | 1.00 [1.00, 1.00] | nan [nan, nan] |
