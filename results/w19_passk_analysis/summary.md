# pass@k analyses (arith+func, 143 gold-gate prompts)

Untrained 360M: 25 prompts correct within 4 masked samples, 35 within 16.

## (a) Subset test: greedy-correct prompts under the full stack vs the base pass@16 set

| student | correct | inside base pass@16 | outside | outside as % of correct |
|---|---|---|---|---|
| base, greedy full stack | 26 | 25 | 1 | 4% |
| SSD | 20 | 19 | 1 | 5% |
| SSD s1 | 20 | 19 | 1 | 5% |
| SSD s2 | 20 | 20 | 0 | 0% |
| RFT | 18 | 18 | 0 | 0% |
| RFT s1 | 18 | 18 | 0 | 0% |
| RFT s2 | 18 | 18 | 0 | 0% |
| SSD + functional reward | 19 | 19 | 0 | 0% |
| SSD + coverage pool | 22 | 20 | 2 | 9% |
| SSD + dense scope mask | 23 | 22 | 1 | 4% |
| SSD + one gold per group | 42 | 24 | 18 | 43% |
| SFT-gold then SSD | 37 | 22 | 15 | 41% |
| SFT-gold | 39 | 18 | 21 | 54% |
| SFT-gold s1 | 45 | 19 | 26 | 58% |
| SFT-gold s2 | 44 | 20 | 24 | 55% |

## (a2) Solved outside the base pass@k set, k = [4, 16, 64, 128, 256]

| student | correct | outside @4 | outside @16 | outside @64 | outside @128 | outside @256 |
|---|---|---|---|---|---|---|
| base, greedy full stack | 26 | 5 | 1 | 1 | 0 | 0 |
| SSD | 20 | 2 | 1 | 0 | 0 | 0 |
| SSD s1 | 20 | 2 | 1 | 1 | 0 | 0 |
| SSD s2 | 20 | 2 | 0 | 0 | 0 | 0 |
| RFT | 18 | 2 | 0 | 0 | 0 | 0 |
| RFT s1 | 18 | 2 | 0 | 0 | 0 | 0 |
| RFT s2 | 18 | 2 | 0 | 0 | 0 | 0 |
| SSD + functional reward | 19 | 1 | 0 | 0 | 0 | 0 |
| SSD + coverage pool | 22 | 4 | 2 | 1 | 1 | 1 |
| SSD + dense scope mask | 23 | 3 | 1 | 0 | 0 | 0 |
| SSD + one gold per group | 42 | 24 | 18 | 17 | 15 | 15 |
| SFT-gold then SSD | 37 | 18 | 15 | 15 | 13 | 13 |
| SFT-gold | 39 | 26 | 21 | 19 | 16 | 16 |
| SFT-gold s1 | 45 | 31 | 26 | 25 | 22 | 22 |
| SFT-gold s2 | 44 | 29 | 24 | 22 | 19 | 19 |

Base pass@k (of 143 gate prompts): k=4: 25, k=16: 35, k=64: 40, k=128: 47, k=256: 49.

## (b) Sampled pass@1 (16 masked samples per prompt, temp 0.8) with bootstrap 95% CI over prompts

| checkpoint | greedy full-stack correct | sampled pass@1 (of 143) [CI] | pass@16 |
|---|---|---|---|
| smollm2-1.7b-instruct | 48 | 30.9 [23.2, 39.1] | 59 |
| smollm2-1.7b-instruct_rft_s0 | 29 | 26.2 [17.5, 35.5] | 27 |
| smollm2-1.7b-instruct_ssd_s0 | 43 | 40.9 [30.9, 51.2] | 49 |
| smollm2-1.7b-instruct_ssd_s1 | 55 | 44.1 [34.1, 54.2] | 55 |
| smollm2-360m-instruct | 26 | 14.1 [9.1, 19.5] | 49 |
| smollm2-360m-instruct_sft_s0 | 39 | 38.6 [29.6, 48.1] | 61 |
| smollm2-360m-instruct_ssd_s0 | 20 | 19.5 [12.3, 27.6] | 27 |
| smollm2-360m-instruct_ssd_s0__gold_in_group=1 | 42 | 28.6 [21.3, 36.3] | 61 |
| smollm3-3b | 55 | 37.1 [28.5, 46.0] | 61 |
| smollm3-3b_rft_s0 | 11 | 6.0 [2.0, 11.0] | 7 |
| smollm3-3b_ssd_s0 | 57 | 54.4 [43.6, 65.9] | 59 |
