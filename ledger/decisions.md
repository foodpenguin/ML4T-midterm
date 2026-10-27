# Decision log

Decisions that change what is run or what is claimed, with the evidence available when they
were made. The decisions are dated. "Before the holdout" means no holdout row had been read
(`ledger/holdout_touches.csv` was empty).

## 2026-10-27: RL operating model (F7) dropped, not run

**Decision**: the planned setup-v2 RL policy for exposure, entry threshold and rebalancing over
out-of-fold predictions is not built. Ledger family F7 is marked `dropped` with 0 trials, so it
does not enter the Deflated Sharpe count.

**Evidence at the time (dev period only)**:
- **Reward dominated by three days.** PnL over 969 days is dominated by three single-name short
  squeezes (2025-04-30 ALPACAUSDT; 2025-09-07/08 MYXUSDT). They cost the ML books 43–45% of
  capital and moved their Sharpe by 0.6–0.8. A policy trained on this one path would mostly learn
  to avoid those three events.
- **Path-dependent controls are already luck.** The registered kill-switch ladder flips between
  net Sharpe −0.15 and +0.22 when costs move by ±2%. An RL policy is a richer version of the
  same kind of control.
- **Too few independent observations.** There are about 140 independent weeks of out-of-fold
  predictions on a single historical path. Synthetic environments are allowed only for
  robustness, never for the headline.
- **The binding risk has a one-parameter fix.** It is position-level tail risk, which a rule
  addresses with zero or one parameter (see the next step below). Drift and overfitting would be
  near-certain for a learned policy in this setting.

## 2026-10-27: Next research step for Section 7 (named, not run before the holdout)

**Decision**: the single named next step is a **setup v2 that removes the short-leg tail
risk**. It is not implemented before the holdout. Setup v1 stays frozen and is the only setup
the holdout tests.

**Why this step**:
- **Where value is lost.** The diagnosis (`reports/models_diagnosis.md`, `reports/backtest.md`,
  `reports/deep.md`) shows signal-stage value (IC ≈ 0.13 for every ML rung) that disappears at
  the portfolio stage. Median Q5−Q1 spreads are strongly positive, while mean spreads are about
  zero.
- **What causes the loss.** The loss sits in the short leg's right tail: young, volatile
  contracts that squeeze by several hundred percent in a day.
- **It was a known risk.** The project owner's earlier research had flagged that shorting small
  contracts carries this risk, and that was the reason the earlier small-cap strategy was never
  traded. Setup v1 did not encode it, because the 10% per-name cap binds at entry only.

**What v2 would change** (a change of mechanics, so a new setup version, not a parameter):
- **Short-leg constraint.** Either:
  - a liquidation-aware cap, so that no short can lose more than its isolated-margin 1×
    liquidation loss (roughly +100% price move); or
  - an eligibility rule that excludes the youngest or most volatile tier from the short leg.
- **Universe and windows (alternative studied in discussion).** Restrict training and trading
  to the most liquid names, for example the top 10 by trailing dollar volume, and use shorter
  windows. It was not chosen as the next step, for two reasons:
  - With 10 names, legs shrink to about 2 names and daily cross-sectional IC loses its power.
  - The idea was formed after seeing which names caused the losses.

  It stays a candidate for a later iteration.

**How v2 would be evaluated**: v2 would be registered before any run. Every variant counts in
the ledger. It would be evaluated on new data (the forward shadow period or a later holdout),
because the current holdout will already have been used once by v1.

## 2026-10-27: Setup v2 code-review fixes (before any shadow data)

**Decision**: fix the engine's v2 accounting and clarify the v2b rule, then rerun the dev
exploration as a bug-fix rerun, not as a trial (F9 stays at 2 trials). No shadow day existed
yet, so nothing here could be shaped by the registered test's data. The amendments are listed in
`config/setup_v2.yaml -> amendments`.
- **Accounting fixes.** A liquidation counts as turnover. A liquidated position earns no funding
  on its liquidation day. `liquidations` counts names, not tranche positions. The shadow test
  counts traded days with frozen predictions, not calendar days.
- **v2b clarification.** A name with unknown volatility is not eligible for the short leg.
- **Guards and structure.** Shadow-period prices can enter PnL only after a logged shadow read in
  the same process. The variant definitions and the registered test now live in
  `src/crowding/setup_v2.py`, not in an exploration script.

## 2026-10-27: Setup v2 Monte Carlo robustness (descriptive; no decision changed)

Registered in `config/setup_v2.yaml -> monte_carlo` (commit `bd73678`) before any simulation;
results in `reports/setup_v2_mc.md` (run R0042). No v2 parameter was changed.
- **Where v2a's cap hurts.** v2a loses to v1 when spikes of 2.5x or more revert and come often.
  If they fully revert, v2a loses from 8 a year on top of the real path (its Sharpe is higher in
  only 8-12% of simulations), and at 24 a year its max drawdown is about 13 pp worse. If half the
  move reverts, v2a still loses at 24 a year with 2.5x spikes (7%). When the move fully persists,
  v2a is better in every simulation.
- **What the shadow test can show.** Resampling the dev period, the registered rule supports v2a
  on 46% of 90-day paths and 57% of 180-day paths. Almost no path has a positive lower bound
  on the return difference, so any support will come from the tail clause.
- **Ordinary days.** Without the three squeeze days, v2a costs 1.4 bp a day
  [-2.9, +0.1] against v1.
