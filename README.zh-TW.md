<p align="center"><img src="aseets/sponsor_card.png" alt="このrepoは、ご覧のClaudeの提供でお送りします" width="720"></p>

<p align="center"><a href="README.md">English</a> | <b>繁體中文</b></p>

# 永續合約擁擠溢價（ML4T 期中專案）

在 Binance USDT 本位永續合約上完整走一遍 ML4T 工作流程：
- **假說**：擁擠的槓桿多頭持續支付資金費、之後被迫平倉，因此資金費與部位資料能排序流動性前 100 大合約的下週報酬。
- **交易合約**：[`setup.yaml`](setup.yaml)，setup v1。
- **封存 holdout**：決策日 2025-10-25 至 2026-10-24，在 commit `38132d0` 封存，只讀過一次（[`ledger/holdout_touches.csv`](ledger/holdout_touches.csv)）。部分事前曝光揭露於 [`ledger/prior_exposure.md`](ledger/prior_exposure.md)。

## 結果一覽

| | 驗證期（開發期，32 個 fold） | Holdout（只讀一次） |
|---|---|---|
| Ridge 每日 IC [HAC 95% CI] | +0.131 [+0.102, +0.160] | **+0.140 [+0.107, +0.173]，確認性檢定通過** |
| Ridge 淨 Sharpe [95% CI] | −0.19 [−1.39, +1.02] | −0.68 [−2.64, +1.28] |

訊號站得住，但沒有轉成報酬：損益屬於 Chapter 20 的「never resolved」。開發期中，三次單一幣種的
軋空就吃掉約 43% 的資金。唯一指名的下一步是 **setup v2**（空頭腿尾部控制），在 holdout 之後才登記，
只用新資料檢驗：[`reports/setup_v2_dev.md`](reports/setup_v2_dev.md)、
[`reports/setup_v2_mc.md`](reports/setup_v2_mc.md)、[`reports/shadow_v2.md`](reports/shadow_v2.md)。

## 目錄結構

| 路徑 | 內容 |
|---|---|
| `setup.yaml` | 交易設定 v1、評估協定、事先登記的確認性檢定 |
| `config/` | 各階段的登記檔，皆在該階段執行前 commit |
| `ledger/` | 研究帳本（試驗家族與預算）、決策紀錄、holdout 讀取紀錄、事前曝光揭露 |
| `run_log.csv` | 每次執行一列：commit、資料快照、設定雜湊、切分日期、指標、決策 |
| `notebooks/` | `01_setup` … `08_setup_v2`：每個階段一本，附中英註解；`07_results` 會重現 headline table |
| `src/crowding/` | 函式庫：資料載入、股票池、標籤、特徵、切分、模型、回測引擎、holdout 與 v2 工具 |
| `scripts/` | 執行入口，一律在 repo 根目錄執行 |
| `tests/` | 單元測試，以及在真實資料上的時點正確性測試 |
| `reports/` | 產出的報告、CSV 與圖 |
| `tools/` | 產生 `data/` 的原始資料蒐集腳本（保留出處） |
| `doc/research/` | 早期探索，已在帳本中計為先前試驗 |
| `data/`、`artifacts/` | 資料快照與模型產物：**未 commit**，請從下方 Drive 下載 |

## 資料與產物（未 commit）

請從共用的 Google Drive 資料夾下載：
**<https://drive.google.com/drive/folders/1xVm0xtljfZASO_C5JyxEC5JCv8ujLIaB?usp=drive_link>**

放在 repo 根目錄，讓以下路徑存在：

| 路徑 | 大小 | 用途 |
|---|---|---|
| `data/processed/`（含 `sealed/`） | 約 90 MB | 所有 notebook 與研究腳本 |
| `data/UNIVERSE/` | 約 110 MB | 日 K 線：所有回測都要用，包含 notebook 02、05、07 與 W5、W6、v2 腳本 |
| `data/METRICS/`、`data/raw/funding_settlements/` | 約 630 MB | 重建資料集，以及 notebook 02 的時點正確性測試 |
| `artifacts/predictions/` | 約 25 MB | notebook 04、05、08 與 v2 腳本 |
| `artifacts/holdout/`（`deep/` 可略過） | 約 15 MB | notebook 07（headline table） |
| `artifacts/deep/` | 約 540 MB | 選用：在 Colab 重新訓練第 4 階深度模型 |

簡單說：下載 `data/`（舊的 `ALPHA/`、`LP/` 不需要）與 `artifacts/`（`artifacts/deep/` 與
`artifacts/holdout/deep/` 不需要）。

`data/processed/manifest.json` 記錄每個輸入檔的 SHA-256，可用來確認你的資料與產生報告時的版本一致。

## 重現

一律在 repo 根目錄執行（Python 3.11 以上）：

```bash
pip install -r requirements.txt
python -m pytest
```

**Notebooks**（最快的檢查方式；需要 Drive 上的 `data/` 與 `artifacts/`，見上表）：

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/07_results.ipynb
```

`07_results` 會從已快取的 holdout 產物重新產生 holdout 報告到 `artifacts/rerun/holdout/`，不讀封存資料、
不寫 run log，再逐格與已 commit 的 `reports/holdout.md` 比對 headline table。其他 notebook 各自即時重算
一個關鍵數字（一個特徵的 IC、Ridge 的 IC、Ridge 回測），並與報告比對。

**完整流程**：從原始快照重建資料集，再依序執行各階段。每個階段都先在 `config/` 登記才執行，並寫入 `run_log.csv`：

```bash
python scripts/download_funding_settlements.py --end 2026-10-25
python scripts/build_dataset.py
python scripts/data_quality_report.py
python scripts/run_triage.py          # W3 特徵篩選        -> reports/triage.md
python scripts/run_models.py          # W4 模型階梯        -> reports/models.md
python scripts/diagnose_models.py     #    W4 診斷         -> reports/models_diagnosis.md
python scripts/run_backtest.py        # W5 回測與風險      -> reports/backtest.md
python scripts/run_deep.py prepare    # 第 4 階（選用），之後在 Colab 執行：見 scripts/colab/README.md
python scripts/run_deep.py evaluate   #                    -> reports/deep.md
HOLDOUT_UNSEAL=1 python scripts/run_holdout.py predict   # W6，唯一一次讀取 holdout
python scripts/run_holdout.py report  #                    -> reports/holdout.md
```

`run_holdout.py predict` 拒絕執行第二次：holdout 只讀一次。確認性結果在 `reports/holdout_confirmatory.json`。

**Setup v2**（空頭腿尾部控制，`config/setup_v2.yaml`），在 holdout 之後：

```bash
python scripts/run_setup_v2.py                    # 開發期探索（探索性）       -> reports/setup_v2_dev.md
python scripts/run_setup_v2_mc.py                 # 蒙地卡羅穩健性（描述性）   -> reports/setup_v2_mc.md
HOLDOUT_UNSEAL=1 python scripts/run_shadow.py     # 事先登記的前瞻檢定         -> reports/shadow_v2.md
```

`run_shadow.py` 需要延長到 2026-10-25 之後的資料快照。不滿 90 個實現日不下任何結論；執行時會先確認凍結的
Ridge（`config/frozen_ridge.json`）能重現已 commit 的 `reports/shadow_predictions.csv`。

`data/` 的原始快照由 `tools/` 中的腳本蒐集：
- `fetch_universe_ext.py`：日 K 線。
- `backfill_metrics.py`：data.binance.vision 的 metrics 資料。
- `daily_update.py`：每日延伸。

要從頭重建，先執行這些腳本，再依 `setup.yaml` 的 `snapshot_end_exclusive` 截斷。

## 重跑時數字可能不同的原因

- **下載日期與資料商修正**：Binance 沒有公布 K 線與資金費歷史的修正政策，已下市的幣也可能隨時間從部分 API
  消失，所以重新下載的資料可能和 Drive 上的快照不同。manifest 的雜湊值可以看出是否不同。
- **隨機種子**：LightGBM 與 Optuna 使用種子 42，蒙地卡羅使用種子 20261027，在相同資料上可完全重現。第 4 階
  深度模型在 Colab A100 上訓練，GPU 運算不保證逐位元一致，重新訓練會得到略有差異的預測；已提交的預測在
  `artifacts/predictions/`。
- **試驗次數**：Deflated Sharpe Ratio 使用讀取 holdout 當時已存在的試驗數（454）。之後才登記的 setup v2
  家族不計入（`trials_run(before_holdout=True)`）。
