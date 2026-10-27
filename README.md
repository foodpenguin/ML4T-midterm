# Perpetual-futures crowding premium (ML4T midterm)

One complete pass through the ML4T workflow on Binance USDT-M perpetuals:
- **Hypothesis**: crowded leveraged longs pay funding and later unwind, so funding and positioning data rank next-week returns across the 100 most liquid contracts.
- **Contract**: [`setup.yaml`](setup.yaml), setup v1.
- **Sealed holdout**: decision dates 2025-10-25 to 2026-10-24, sealed in commit `38132d0`. Its partial prior exposure is disclosed in [`ledger/prior_exposure.md`](ledger/prior_exposure.md).

## Layout

| Path | What it is |
|---|---|
| `setup.yaml` | Trading setup v1, evaluation protocol, registered confirmatory test |
| `ledger/` | Research ledger (trial families and budgets), prior-exposure disclosure |
| `run_log.csv` | One row per run: commit, data snapshot, config hash, split dates, metrics, decision |
| `src/crowding/` | Library: loaders (`io`), universe, labels, features, quality checks |
| `scripts/` | Entry points, run from the repository root |
| `tests/` | Point-in-time and label-timing tests |
| `reports/` | Generated reports and figures (data quality, ...) |
| `tools/` | Raw-data collectors that produced `data/` (legacy, kept for provenance) |
| `doc/research/` | Earlier exploration, counted as prior trials in the ledger |
| `data/` | Raw snapshot and `data/processed/`, **not committed** |

## Reproduce

Run everything from the repository root (Python 3.11+):

```bash
pip install -r requirements.txt
python scripts/download_funding_settlements.py --end 2026-10-25
python scripts/build_dataset.py
python scripts/data_quality_report.py
python -m pytest
```

The research pipeline after the data build, in order (each step is registered in `config/`
before it runs, and logs to `run_log.csv`):

```bash
python scripts/run_triage.py          # W3 feature triage      -> reports/triage.md
python scripts/run_models.py          # W4 baseline ladder     -> reports/models.md
python scripts/diagnose_models.py     #    W4 diagnosis        -> reports/models_diagnosis.md
python scripts/run_backtest.py        # W5 backtest and risk   -> reports/backtest.md
python scripts/run_deep.py prepare    # rung 4 (optional), then Colab: see scripts/colab/README.md
python scripts/run_deep.py evaluate   #                        -> reports/deep.md
HOLDOUT_UNSEAL=1 python scripts/run_holdout.py predict   # W6, the single holdout touch
python scripts/run_holdout.py report  #                        -> reports/holdout.md
```

`run_holdout.py predict` refuses to run twice: the holdout is read once
(`ledger/holdout_touches.csv`). Its confirmatory result is in `reports/holdout_confirmatory.json`.

The raw snapshot in `data/` was collected with the scripts in `tools/`:
- `fetch_universe_ext.py`: daily klines.
- `backfill_metrics.py`: data.binance.vision metrics.
- `daily_update.py`: daily extension.

To rebuild it from scratch, run those scripts and then cut the result at `snapshot_end_exclusive` in `setup.yaml`.

**Why numbers can differ on a rerun**:
- Binance documents no revision policy for klines or funding history.
- Delisted symbols can drop out of some endpoints over time, so a later download may have fewer leavers.
- `data/processed/manifest.json` records the SHA-256 of every input, so a mismatch is detectable.
