# Prior exposure before the holdout was sealed

The holdout (decision dates 2025-10-25 to 2026-10-24) was sealed on 2026-10-26, but some of
it had already been seen. This project grew out of exploratory scripts that ran over the full
history before any protocol existed. This file lists everything those scripts touched, so the
holdout result can be read with that in mind. The counts feed the trial total used for the
Deflated Sharpe Ratio (`ledger/research_ledger.csv`, family `F0_prior`).

| Script (date) | Family | What was tested | Period seen | Overlap with holdout | Trials |
|---|---|---|---|---|---|
| `data/UNIVERSE/xs_explore_results.csv`, `tools/explore_xs.py` (2026-08-29) | price momentum / reversal | 8 signals × 3 liquidity tiers × 3 horizons, non-overlapping long-short t-tests | 2021 → 2026-08 | about 10 of 12 months | 72 |
| `data/UNIVERSE/xs_explore_v2.csv`, `tools/explore_xs2.py` (2026-08-29) | price momentum / reversal | 7 lookbacks × 3 tiers × 3 horizons | 2021 → 2026-08 | about 10 of 12 months | 63 |
| `doc/research/results/xs_cap20000_331dd54b.csv` (2026-08-29) | price momentum / reversal | 8 signals × 4 tiers × 3 horizons; OOS = last 30% | 2019 → 2026-08 | about 10 of 12 months | 96 |
| `doc/research/results/smallcap_sensitivity.csv` (2026-08-29) | small-cap momentum + reversal | 12 configurations of a traded system | 2023-07 → 2026-07 | about 9 of 12 months | 12 |
| `tools/run_momentum.py` (2026-07-14) | price momentum | top-50 long-short, lookback × rebalance grid | 2021 → 2026-07-13 | about 8.6 of 12 months | 16 |
| **`tools/run_carry.py` (2026-07-14)** | **funding carry** | **top-50 long-short on −(7-day mean funding), 20% legs, 7-day rebalance; plus 8 cash-and-carry configurations** | 2021 → 2026-07-13 | **about 8.6 of 12 months** | 9 |
| **`tools/analyze_alpha.py` (2026-08-24)** | **positioning** | **OI change (6/12/24h), top-position vs global-account L/S divergence (level, 12/24h change), top-position L/S, global-account L/S, taker buy/sell ratio, each at 4/12/24/48h horizons on hourly data** | 2026-07-01 → 2026-08-23 | about 1.8 of 12 months | 36 |
| `doc/research/FINDINGS.md` LP section (2026-08-29) | DEX liquidity provision | 27 pools | 2022 → 2025-08 | none | 27 |

**Total prior trials: 331.**

## What this does and does not contaminate

- **Directly exposed in the holdout**:
  - The rule that is now rung 1 of the ladder, `rule_carry` (cross-sectional −funding_7d). Its spec differs from the old rule: a top-100 universe instead of top-50, settlement-exact funding, daily staggered tranches instead of a weekly rebalance. But it is the same idea. **Its holdout number is not confirmatory evidence**, and it is reported as a baseline only.
  - A handful of positioning signals, on about 1.8 months of the holdout at hourly horizons.
- **Not exposed**: the feature set as a whole, the daily 7-day label, the top-100 universe, and every fitted model (Ridge, LightGBM, rung 4–5). The confirmatory test is on the deployed model's IC, not on any of the old rules.
- **Bias direction**: if the old results shaped which mechanism was chosen, the holdout IC is biased upward. The mechanism was chosen from Table 6.1 and from data availability, without re-reading those outputs. But the person who ran the scripts saw their results, so the bias cannot be ruled out.
- **Mitigation**:
  - The decision dates from 2026-10-25 on form a **forward shadow period** that no one has seen. It is reported separately from the holdout.
  - The 331 prior trials are added to the trial count for the Deflated Sharpe Ratio.
