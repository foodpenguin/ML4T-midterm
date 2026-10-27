<p align="center"><img src="aseets/sponsor_card.png" alt="このrepoは、ご覧のClaudeの提供でお送りします" width="720"></p>

<p align="center"><b>English</b> | <a href="README.zh-TW.md">繁體中文</a></p>

# Perpetual-futures crowding premium (ML4T midterm)

One complete pass through the ML4T workflow on Binance USDT-M perpetuals:
- **Hypothesis**: crowded leveraged longs pay funding and later unwind, so funding and positioning data rank next-week returns across the 100 most liquid contracts.
- **Contract**: [`setup.yaml`](setup.yaml), setup v1.
- **Sealed holdout**: decision dates 2025-10-25 to 2026-10-24, sealed in commit `38132d0` and read once ([`ledger/holdout_touches.csv`](ledger/holdout_touches.csv)). Its partial prior exposure is disclosed in [`ledger/prior_exposure.md`](ledger/prior_exposure.md).

## Results at a glance

| | Validation (dev, 32 folds) | Holdout (one touch) |
|---|---|---|
| Ridge daily IC [HAC 95% CI] | +0.131 [+0.102, +0.160] | **+0.140 [+0.107, +0.173], confirmatory test PASS** |
| Ridge net Sharpe [95% CI] | −0.19 [−1.39, +1.02] | −0.68 [−2.64, +1.28] |

The signal holds, but it does not turn into returns: the PnL is "never resolved" (Chapter 20).
Three single-name short squeezes cost the book about 43% of capital on dev. The one named next
step is **setup v2** (short-leg tail control). It was registered after the holdout and is
tested only on new data: [`reports/setup_v2_dev.md`](reports/setup_v2_dev.md),
[`reports/setup_v2_mc.md`](reports/setup_v2_mc.md), [`reports/shadow_v2.md`](reports/shadow_v2.md).

## Layout

| Path | What it is |
|---|---|
| `setup.yaml` | Trading setup v1, evaluation protocol, registered confirmatory test |
| `config/` | Registrations of each stage, committed before the stage ran |
| `ledger/` | Research ledger (trial families and budgets), decision log, holdout touch log, prior-exposure disclosure |
| `run_log.csv` | One row per run: commit, data snapshot, config hash, split dates, metrics, decision |
| `notebooks/` | `01_setup` … `08_setup_v2`: one per stage, with English and Chinese notes; `07_results` reproduces the headline table |
| `src/crowding/` | Library: data loaders, universe, labels, features, splits, models, backtest engine, holdout and v2 helpers |
| `scripts/` | Entry points, run from the repository root |
| `tests/` | Unit tests and point-in-time tests on the real data |
| `reports/` | Generated reports, CSVs and figures |
| `tools/` | Raw-data collectors that produced `data/` (kept for provenance) |
| `doc/research/` | Earlier exploration, counted as prior trials in the ledger |
| `data/`, `artifacts/` | Data snapshot and model outputs: **not committed**, download them from Drive (below) |

## Data and artifacts (not committed)

Download them from the shared Google Drive folder:
**<https://drive.google.com/drive/folders/1xVm0xtljfZASO_C5JyxEC5JCv8ujLIaB?usp=drive_link>**

Put the folders at the repository root so that these paths exist:

| Path | Size | Needed for |
|---|---|---|
| `data/processed/` (with `sealed/`) | ~90 MB | every notebook and research script |
| `data/UNIVERSE/` | ~110 MB | daily bars: every backtest, so notebooks 02, 05, 07 and the W5, W6 and v2 scripts |
| `data/METRICS/`, `data/raw/funding_settlements/` | ~630 MB | rebuilding the dataset, and the point-in-time tests in notebook 02 |
| `artifacts/predictions/` | ~25 MB | notebooks 04, 05, 08 and the v2 scripts |
| `artifacts/holdout/` (`deep/` can be skipped) | ~15 MB | notebook 07 (headline table) |
| `artifacts/deep/` | ~540 MB | optional: retraining the rung-4 deep model on Colab |

In short: download `data/` (except the legacy `ALPHA/` and `LP/` folders) and `artifacts/`
(except `artifacts/deep/` and `artifacts/holdout/deep/`).

`data/processed/manifest.json` records the SHA-256 of every input file, so you can check that
your copy matches the one the reports were built from.

## Reproduce

Run everything from the repository root (Python 3.11+):

```bash
pip install -r requirements.txt
python -m pytest
```

**Notebooks** (fastest check; needs `data/` and `artifacts/` from Drive, see the table above):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/07_results.ipynb
```

`07_results` regenerates the holdout report from the cached holdout artifacts into
`artifacts/rerun/holdout/`. It does not read the sealed data and writes no run-log row. It then
compares the headline table cell by cell with the committed `reports/holdout.md`. The other
notebooks recompute one key number each (a triage IC, the ridge IC, the ridge backtest) and
compare it with the report.

**Full pipeline**: rebuild the dataset from the raw snapshot, then run every stage in order.
Each stage is registered in `config/` before it runs and logs to `run_log.csv`:

```bash
python scripts/download_funding_settlements.py --end 2026-10-25
python scripts/build_dataset.py
python scripts/data_quality_report.py
python scripts/run_triage.py          # W3 feature triage      -> reports/triage.md
python scripts/run_models.py          # W4 baseline ladder     -> reports/models.md
python scripts/diagnose_models.py     #    W4 diagnosis        -> reports/models_diagnosis.md
python scripts/run_backtest.py        # W5 backtest and risk   -> reports/backtest.md
python scripts/run_deep.py prepare    # rung 4 (optional), then Colab: see scripts/colab/README.md
python scripts/run_deep.py evaluate   #                        -> reports/deep.md
HOLDOUT_UNSEAL=1 python scripts/run_holdout.py predict   # W6, the single holdout touch
python scripts/run_holdout.py report  #                        -> reports/holdout.md
```

`run_holdout.py predict` refuses to run twice: the holdout is read once. Its confirmatory
result is in `reports/holdout_confirmatory.json`.

**Setup v2** (short-leg tail control, `config/setup_v2.yaml`), after the holdout:

```bash
python scripts/run_setup_v2.py                    # dev exploration (exploratory)        -> reports/setup_v2_dev.md
python scripts/run_setup_v2_mc.py                 # Monte Carlo robustness (descriptive) -> reports/setup_v2_mc.md
HOLDOUT_UNSEAL=1 python scripts/run_shadow.py     # registered forward test              -> reports/shadow_v2.md
```

`run_shadow.py` needs a snapshot extended past 2026-10-25. It concludes nothing before 90
realized days, and it first checks that the frozen ridge (`config/frozen_ridge.json`)
reproduces the committed `reports/shadow_predictions.csv`.

The raw snapshot in `data/` was collected with the scripts in `tools/`:
- `fetch_universe_ext.py`: daily klines.
- `backfill_metrics.py`: data.binance.vision metrics.
- `daily_update.py`: daily extension.

To rebuild it from scratch, run those scripts and then cut the result at
`snapshot_end_exclusive` in `setup.yaml`.

## Why numbers can differ on a rerun

- **Download date and vendor revisions.** Binance documents no revision policy for klines or
  funding history, and delisted symbols can drop out of some endpoints over time. A fresh
  download can therefore differ from the Drive snapshot. The manifest hashes show whether it does.
- **Random seeds.** LightGBM and Optuna use seed 42, and the Monte Carlo uses seed 20261027, so
  they repeat exactly on the same data. The rung-4 deep model was trained on a Colab A100: GPU
  kernels are not bit-for-bit deterministic, so retraining it gives slightly different
  predictions. Its committed predictions are in `artifacts/predictions/`.
- **Trial count.** The Deflated Sharpe Ratio counts the trials that existed when the holdout was
  touched (454). Setup v2 families registered later are excluded (`trials_run(before_holdout=True)`).
