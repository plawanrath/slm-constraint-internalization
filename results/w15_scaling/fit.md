# Scaling fit: Residual_k(size) = a · size^(−β)

Paired prompt-resampling bootstrap on β (2000 resamples, seed 0); residuals ≤ 0.001 clipped before the log. Sizes = exact parameter counts of the pinned checkpoints. `base` = free-vs-masked violation gap of the untrained model (Table 1); `trained:<method>` = residual after training (seed-averaged). A fit is `degenerate` when all but one point are clipped (the layer has no positive residual at these scales).

| series | family | pool | layer | n | β | β 95% CI | a | R² | clipped | points (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | smollm | arith_func_200 | v1 | 4 | 0.083 | [0.061, 0.107] | 4.79 | 0.85 | 0 | smollm2-135m-instruct=+100.0, smollm2-360m-instruct=+98.5, smollm2-1.7b-instruct=+78.0, smollm3-3b=+81.5 |
| base | smollm | arith_func_200 | v2 | 4 | -0.645 | [-0.974, -0.000] | 3.83e-09 | 0.46 | 3 (degenerate) | smollm2-135m-instruct=-0.5, smollm2-360m-instruct=-2.0, smollm2-1.7b-instruct=-1.0, smollm3-3b=+1.5 |
| base | smollm | arith_func_200 | v3 | 4 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 4 (degenerate) | smollm2-135m-instruct=-13.0, smollm2-360m-instruct=-14.0, smollm2-1.7b-instruct=-22.0, smollm3-3b=-26.0 |
| base | smollm | arith_func_200 | v4 | 4 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 4 (degenerate) | smollm2-135m-instruct=-1.0, smollm2-360m-instruct=-7.5, smollm2-1.7b-instruct=-10.0, smollm3-3b=-8.0 |
| base | smollm | linalg_125 | v1 | 4 | 0.231 | [0.185, 0.283] | 77.7 | 0.55 | 0 | smollm2-135m-instruct=+100.0, smollm2-360m-instruct=+96.8, smollm2-1.7b-instruct=+38.4, smollm3-3b=+66.4 |
| base | smollm | linalg_125 | v2 | 4 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 4 (degenerate) | smollm2-135m-instruct=+0.0, smollm2-360m-instruct=+0.0, smollm2-1.7b-instruct=+0.0, smollm3-3b=+0.0 |
| base | smollm | linalg_125 | v3 | 4 | -0.000 | [-0.495, -0.000] | 0.001 | 1.00 | 4 (degenerate) | smollm2-135m-instruct=-15.2, smollm2-360m-instruct=-4.8, smollm2-1.7b-instruct=-2.4, smollm3-3b=-2.4 |
| base | smollm | linalg_125 | v4 | 4 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 4 (degenerate) | smollm2-135m-instruct=-1.6, smollm2-360m-instruct=-8.8, smollm2-1.7b-instruct=-3.2, smollm3-3b=-17.6 |
| base | smollm | all | v1 | 4 | 0.130 | [0.108, 0.154] | 11.8 | 0.71 | 0 | smollm2-135m-instruct=+100.0, smollm2-360m-instruct=+97.8, smollm2-1.7b-instruct=+62.8, smollm3-3b=+75.7 |
| base | smollm | all | v2 | 4 | -0.529 | [-0.812, -0.000] | 3.58e-08 | 0.46 | 3 (degenerate) | smollm2-135m-instruct=-0.3, smollm2-360m-instruct=-1.2, smollm2-1.7b-instruct=-0.6, smollm3-3b=+0.9 |
| base | smollm | all | v3 | 4 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 4 (degenerate) | smollm2-135m-instruct=-13.8, smollm2-360m-instruct=-10.5, smollm2-1.7b-instruct=-14.5, smollm3-3b=-16.9 |
| base | smollm | all | v4 | 4 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 4 (degenerate) | smollm2-135m-instruct=-1.2, smollm2-360m-instruct=-8.0, smollm2-1.7b-instruct=-7.4, smollm3-3b=-11.7 |
| base | gemma | arith_func_200 | v1 | 2 | -0.360 | [-0.478, -0.256] | 0.000458 | 1.00 | 0 | gemma-3-270m-it=+49.5, gemma-3-1b-it=+79.5 |
| base | gemma | arith_func_200 | v2 | 2 | 1.223 | [-1.749, 2.057] | 1.01e+08 | 1.00 | 1 (degenerate) | gemma-3-270m-it=+0.5, gemma-3-1b-it=+0.0 |
| base | gemma | arith_func_200 | v3 | 2 | 0.000 | [0.000, 0.000] | 0.001 | 1.00 | 2 (degenerate) | gemma-3-270m-it=-9.0, gemma-3-1b-it=-33.5 |
| base | gemma | arith_func_200 | v4 | 2 | 0.000 | [0.000, 0.000] | 0.001 | 1.00 | 2 (degenerate) | gemma-3-270m-it=+0.0, gemma-3-1b-it=+0.0 |
| base | gemma | linalg_125 | v1 | 2 | 0.014 | [-0.062, 0.090] | 1.14 | 1.00 | 0 | gemma-3-270m-it=+86.4, gemma-3-1b-it=+84.8 |
| base | gemma | linalg_125 | v2 | 2 | 0.000 | [0.000, 0.000] | 0.001 | 1.00 | 2 (degenerate) | gemma-3-270m-it=+0.0, gemma-3-1b-it=+0.0 |
| base | gemma | linalg_125 | v3 | 2 | 0.000 | [0.000, 0.000] | 0.001 | 1.00 | 2 (degenerate) | gemma-3-270m-it=-4.8, gemma-3-1b-it=-2.4 |
| base | gemma | linalg_125 | v4 | 2 | 0.000 | [0.000, 0.000] | 0.001 | 1.00 | 2 (degenerate) | gemma-3-270m-it=-16.0, gemma-3-1b-it=-12.8 |
| base | gemma | all | v1 | 2 | -0.188 | [-0.256, -0.120] | 0.0167 | 1.00 | 0 | gemma-3-270m-it=+63.7, gemma-3-1b-it=+81.5 |
| base | gemma | all | v2 | 2 | 0.854 | [-1.380, 1.688] | 4.84e+04 | 1.00 | 1 (degenerate) | gemma-3-270m-it=+0.3, gemma-3-1b-it=+0.0 |
| base | gemma | all | v3 | 2 | 0.000 | [0.000, 0.000] | 0.001 | 1.00 | 2 (degenerate) | gemma-3-270m-it=-7.4, gemma-3-1b-it=-21.5 |
| base | gemma | all | v4 | 2 | 0.000 | [0.000, 0.000] | 0.001 | 1.00 | 2 (degenerate) | gemma-3-270m-it=-6.2, gemma-3-1b-it=-4.9 |
| base | all | arith_func_200 | v1 | 6 | 0.007 | [-0.018, 0.032] | 0.906 | 0.00 | 0 | smollm2-135m-instruct=+100.0, gemma-3-270m-it=+49.5, smollm2-360m-instruct=+98.5, gemma-3-1b-it=+79.5, smollm2-1.7b-instruct=+78.0, smollm3-3b=+81.5 |
| base | all | arith_func_200 | v2 | 6 | -0.399 | [-0.868, 0.112] | 6.3e-07 | 0.17 | 4 | smollm2-135m-instruct=-0.5, gemma-3-270m-it=+0.5, smollm2-360m-instruct=-2.0, gemma-3-1b-it=+0.0, smollm2-1.7b-instruct=-1.0, smollm3-3b=+1.5 |
| base | all | arith_func_200 | v3 | 6 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 6 (degenerate) | smollm2-135m-instruct=-13.0, gemma-3-270m-it=-9.0, smollm2-360m-instruct=-14.0, gemma-3-1b-it=-33.5, smollm2-1.7b-instruct=-22.0, smollm3-3b=-26.0 |
| base | all | arith_func_200 | v4 | 6 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 6 (degenerate) | smollm2-135m-instruct=-1.0, gemma-3-270m-it=+0.0, smollm2-360m-instruct=-7.5, gemma-3-1b-it=+0.0, smollm2-1.7b-instruct=-10.0, smollm3-3b=-8.0 |
| base | all | linalg_125 | v1 | 6 | 0.212 | [0.168, 0.260] | 54.9 | 0.50 | 0 | smollm2-135m-instruct=+100.0, gemma-3-270m-it=+86.4, smollm2-360m-instruct=+96.8, gemma-3-1b-it=+84.8, smollm2-1.7b-instruct=+38.4, smollm3-3b=+66.4 |
| base | all | linalg_125 | v2 | 6 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 6 (degenerate) | smollm2-135m-instruct=+0.0, gemma-3-270m-it=+0.0, smollm2-360m-instruct=+0.0, gemma-3-1b-it=+0.0, smollm2-1.7b-instruct=+0.0, smollm3-3b=+0.0 |
| base | all | linalg_125 | v3 | 6 | -0.000 | [-0.457, -0.000] | 0.001 | 1.00 | 6 (degenerate) | smollm2-135m-instruct=-15.2, gemma-3-270m-it=-4.8, smollm2-360m-instruct=-4.8, gemma-3-1b-it=-2.4, smollm2-1.7b-instruct=-2.4, smollm3-3b=-2.4 |
| base | all | linalg_125 | v4 | 6 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 6 (degenerate) | smollm2-135m-instruct=-1.6, gemma-3-270m-it=-16.0, smollm2-360m-instruct=-8.8, gemma-3-1b-it=-12.8, smollm2-1.7b-instruct=-3.2, smollm3-3b=-17.6 |
| base | all | all | v1 | 6 | 0.081 | [0.058, 0.106] | 4.11 | 0.23 | 0 | smollm2-135m-instruct=+100.0, gemma-3-270m-it=+63.7, smollm2-360m-instruct=+97.8, gemma-3-1b-it=+81.5, smollm2-1.7b-instruct=+62.8, smollm3-3b=+75.7 |
| base | all | all | v2 | 6 | -0.351 | [-0.733, 0.137] | 1.4e-06 | 0.20 | 4 | smollm2-135m-instruct=-0.3, gemma-3-270m-it=+0.3, smollm2-360m-instruct=-1.2, gemma-3-1b-it=+0.0, smollm2-1.7b-instruct=-0.6, smollm3-3b=+0.9 |
| base | all | all | v3 | 6 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 6 (degenerate) | smollm2-135m-instruct=-13.8, gemma-3-270m-it=-7.4, smollm2-360m-instruct=-10.5, gemma-3-1b-it=-21.5, smollm2-1.7b-instruct=-14.5, smollm3-3b=-16.9 |
| base | all | all | v4 | 6 | -0.000 | [-0.000, -0.000] | 0.001 | 1.00 | 6 (degenerate) | smollm2-135m-instruct=-1.2, gemma-3-270m-it=-6.2, smollm2-360m-instruct=-8.0, gemma-3-1b-it=-4.9, smollm2-1.7b-instruct=-7.4, smollm3-3b=-11.7 |
| trained:ssd | smollm | arith_func_200 | v1 | 1 | – | – | – | – | – | smollm2-360m-instruct=+15.0 (no fit: one scale) |
| trained:ssd | smollm | arith_func_200 | v2 | 1 | – | – | – | – | – | smollm2-360m-instruct=+1.0 (no fit: one scale) |
| trained:ssd | smollm | arith_func_200 | v3 | 1 | – | – | – | – | – | smollm2-360m-instruct=+4.5 (no fit: one scale) |
| trained:ssd | smollm | arith_func_200 | v4 | 1 | – | – | – | – | – | smollm2-360m-instruct=-1.5 (no fit: one scale) |
| trained:ssd | smollm | linalg_125 | v1 | 1 | – | – | – | – | – | smollm2-360m-instruct=+43.2 (no fit: one scale) |
| trained:ssd | smollm | linalg_125 | v2 | 1 | – | – | – | – | – | smollm2-360m-instruct=+0.0 (no fit: one scale) |
| trained:ssd | smollm | linalg_125 | v3 | 1 | – | – | – | – | – | smollm2-360m-instruct=+0.8 (no fit: one scale) |
| trained:ssd | smollm | linalg_125 | v4 | 1 | – | – | – | – | – | smollm2-360m-instruct=-9.6 (no fit: one scale) |
| trained:ssd | smollm | all | v1 | 1 | – | – | – | – | – | smollm2-360m-instruct=+25.8 (no fit: one scale) |
| trained:ssd | smollm | all | v2 | 1 | – | – | – | – | – | smollm2-360m-instruct=+0.6 (no fit: one scale) |
| trained:ssd | smollm | all | v3 | 1 | – | – | – | – | – | smollm2-360m-instruct=+3.1 (no fit: one scale) |
| trained:ssd | smollm | all | v4 | 1 | – | – | – | – | – | smollm2-360m-instruct=-4.6 (no fit: one scale) |
| trained:ssd | all | arith_func_200 | v1 | 1 | – | – | – | – | – | smollm2-360m-instruct=+15.0 (no fit: one scale) |
| trained:ssd | all | arith_func_200 | v2 | 1 | – | – | – | – | – | smollm2-360m-instruct=+1.0 (no fit: one scale) |
| trained:ssd | all | arith_func_200 | v3 | 1 | – | – | – | – | – | smollm2-360m-instruct=+4.5 (no fit: one scale) |
| trained:ssd | all | arith_func_200 | v4 | 1 | – | – | – | – | – | smollm2-360m-instruct=-1.5 (no fit: one scale) |
| trained:ssd | all | linalg_125 | v1 | 1 | – | – | – | – | – | smollm2-360m-instruct=+43.2 (no fit: one scale) |
| trained:ssd | all | linalg_125 | v2 | 1 | – | – | – | – | – | smollm2-360m-instruct=+0.0 (no fit: one scale) |
| trained:ssd | all | linalg_125 | v3 | 1 | – | – | – | – | – | smollm2-360m-instruct=+0.8 (no fit: one scale) |
| trained:ssd | all | linalg_125 | v4 | 1 | – | – | – | – | – | smollm2-360m-instruct=-9.6 (no fit: one scale) |
| trained:ssd | all | all | v1 | 1 | – | – | – | – | – | smollm2-360m-instruct=+25.8 (no fit: one scale) |
| trained:ssd | all | all | v2 | 1 | – | – | – | – | – | smollm2-360m-instruct=+0.6 (no fit: one scale) |
| trained:ssd | all | all | v3 | 1 | – | – | – | – | – | smollm2-360m-instruct=+3.1 (no fit: one scale) |
| trained:ssd | all | all | v4 | 1 | – | – | – | – | – | smollm2-360m-instruct=-4.6 (no fit: one scale) |

No synthetic fit (`results/w08_synth_law/fit.json` missing).
