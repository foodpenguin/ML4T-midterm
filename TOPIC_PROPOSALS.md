# 期中題目評估：根據既有資料與先前研究

> 依據：`MIDTERM_REQUIREMENTS.md`（期中要求）、`data/DATA_QUALITY.md`、`doc/research/FINDINGS.md`、`doc/research/SMALLCAP_BACKTEST.md`、`doc/research/results/*.csv`、課本 Ch. 1–3。
> 撰寫日：2026-10-26。資料截止 2026-10-24（日線、funding、METRICS 均同）。
> **刻意沒有做的事**：這次沒有在任何新特徵上計算報酬或 IC。期中規定「看到結果再改設定就算一次 trial」，而且 holdout 要在交 setup 那天封存。題目是根據**機制 + 資料可用性**挑的，不是看哪個回測好看。

---

## 1. 手上有什麼

| 資料 | 內容 | 期間 | 能不能當主資料 |
|---|---|---|---|
| `UNIVERSE/daily_ohlcv_ext.csv` | Binance USDT-M 永續日線，670 幣（含 13 檔已下市） | 2019-09-08 ~ 2026-10-24 | ✅ 價格骨幹 |
| `UNIVERSE/funding_daily.csv` | 每日 funding 加總（8h/4h 結算） | 2019-09-10 ~ 2026-10-24 | ✅ 覆蓋 99.9% |
| `METRICS/hourly/*.parquet` | 每小時 OI、大戶帳戶/持倉多空比、全市場帳戶多空比、taker 買賣比；已換算成**可知時間**（`snap_lag=0` 佔 99.98%） | 可用橫截面**自 2021-12 起**（118 檔起跳，2025 年約 270–340 檔/小時） | ✅ 約 4.9 年、1,050 萬列 |
| `UNIVERSE/universe_pit.csv` | PIT 前 50 名（`shift(1)`），含已離開者 | 2021-06 ~ 2026-10 | ⚠️ 固定 50 檔，需要時可用 `_ext` 重建 |
| `ALPHA/*` | 即時收集器：小時 OHLCV、OI、多空比、taker、premium/basis、跨所 funding | **2026-07 起**（約 4 個月） | ❌ 太短，不夠 12–24 個月 holdout；**適合拿來寫第 8 節的 parity / shadow mode** |
| `LP/*` | DeFiLlama 102 個池的 APY / IL | 2022 ~ 2025-08 | ❌ `il7d` 缺 75%、沒有池內價格、沒有 PIT 宇宙 |

**優點（對應評分）**：資料品質報告已經很完整。重複 0、OHLC 違反 0、未收盤 K 線已移除並留稽核檔、下市幣有納入、代號重複使用與改名都有揭露、API 和 archive 兩個來源有交叉驗證。第 3 節（15 分）大概已完成 70%。

**缺口**：
1. `DATA_QUALITY.md` 引用的 `tools/check_integrity.py`、`tools/daily_update.py`、`tools/analyze_alpha.py` **不在這個資料夾裡**。重現性測試要求「從 repo 根目錄重建資料」，這些腳本要一起放進 repo。
2. 缺 `shap`、`optuna`、`mapie`（conformal 可以自己寫），`lightgbm`、`sklearn`、`statsmodels` 已裝。
3. 沒有 git repo。

---

## 2. 關鍵限制：先前研究已經「看過」最近兩年

| 先前研究 | 看過的期間 | 測過的訊號 | 試驗數 |
|---|---|---|---|
| `xs_explore_results` / `_v2` | 全期 | 價格動量 21–126d、反轉 3–21d | 72 + 63 |
| `xs_cap20000_331dd54b` | 全期，OOS = 最後 30% | `ret_*`、`volmom_10d/21d` × 4 層 | 96 |
| `SMALLCAP_BACKTEST` | 2023-07 ~ 2026-07，OOS 2025-08 ~ 2026-07 | 小市值動量+反轉 | 12 組敏感度 |
| `FINDINGS` 方向二 | 2022 ~ 2025-08 | LP 池淨收益 | 27 池 |

**結論**：
- **價格動量/反轉這個 trial family 在最近 12–24 個月已經被看過，而且是挑過的**（例如 `volmom_21d` 就是從 96 個檢定中選出、再用 2025–2026 OOS 確認）。拿它當主假說再封存「最後 12 個月」當 holdout，就是課本說的 **holdout contamination，總分上限 60**。
- **funding / OI / 多空比 / taker 這幾個 family 在這個資料夾的研究紀錄裡沒有出現在任何報酬檢定中**，所以可以誠實地封存一段乾淨的 holdout。
  - ⚠️ 例外：`DATA_QUALITY.md` 提到 `tools/analyze_alpha.py` 曾經用 METRICS 特徵預測報酬。**請先確認它分析了哪些特徵、哪段期間**。如果它在最近 12 個月看過 taker 或多空比的報酬，那段要從 holdout 排除，或在帳本揭露。
- 先前研究不是浪費：它可以當作附錄裡**誠實的前期 trial 紀錄**（231+ 個檢定，研究過程中修正成本模型高估 6 倍、IL 符號錯誤），也證明你懂 DSR 為什麼要算 trial 數（研究中隨機訊號的 t 值最高到 2.81）。

---

## 3. 推薦題目

### ★ 題目 A（推薦）：永續合約的擁擠溢酬：用 funding 與倉位資料預測橫截面報酬

**一句話**：在 Binance 永續 PIT 流動性前 100 名中，以 funding、OI 變化、大戶與散戶多空比的分歧預測未來 H 天「價格 + funding」的總報酬，做市場中性 long-short。

**機制（第 1 節兩句話）**
> 槓桿多頭擁擠時，永續價格高於現貨，多方必須付 funding 給空方。願意承接另一邊的人（做空擁擠幣的套利者）承擔了軋空與保證金風險，因此應該拿到補償。當擁擠程度由散戶推動，而大戶站在另一邊時，擁擠更可能以價格回落的方式結束。

- **Edge 來源（Table 6.1）**：funding constraints（主）+ risk compensation + flow predictability（taker 不平衡）。
- **對手是誰**：受保證金和槓桿限制、需要方向性曝險的散戶多頭；交易所 funding 機制強制他們付租金。
- **證偽條件（範例）**：驗證期 funding 特徵的 fold-level Spearman IC 的 HAC 95% CI 包含 0；或扣除 funding 後的純價格報酬 IC 和含 funding 的 IC 方向相反，且 net carry 撐不過 2 × 來回成本；或 regime 三分位中符號翻轉 ≥ 2 次。

**為什麼最適合**
| 期中要求 | 這個題目的對應 |
|---|---|
| Mechanism-first | 課本 Table 6.1 的 funding constraints 幾乎就是這個機制；期中說明列出 `crypto_perps_funding` case study，課本 Ch. 2 也提到用 Binance 永續 19 檔（2020–2025），兩者應該是同一份資料（待確認）。可以寫成「從課本 setup 起步、擴大成 PIT 100 檔」 |
| 乾淨的 holdout | 新的 feature family，可封存 **2025-10-25 ~ 2026-10-24（12 個月）** |
| 10–30 個特徵、多數有機制 | 見下表 |
| 橫截面 → 每日 Spearman IC 有意義 | 每天 100 檔 |
| 資料 PIT 已處理 | METRICS 已換算成可知時間、下市幣有納入 |
| 第 8 節 parity plan | `ALPHA/` 即時收集器就是 live pipeline，而且已和 archive 逐點比對過（OI 99.95% 一致、taker 定義不同不可拼接），可以直接寫進 parity / shadow mode |

**候選特徵（約 15–20 個，需在 W3 做 triage）**
| Feature | Role | 機制 | 備註 |
|---|---|---|---|
| `funding_7d`、`funding_30d`（累積） | Signal | Carry / 擁擠 | 也是 label 的一部分，要小心定義 |
| `funding_z_90d` | Signal | 相對自己歷史的擁擠 | |
| `funding_chg_7d` | Signal | 擁擠加速 | |
| `oi_chg_1d/7d`（vol-scaled） | Signal | 新槓桿進場 | |
| `oi_to_volume` | State | 槓桿深度 | |
| `top_pos_lsr − global_acct_lsr` | Signal | 大戶 vs 散戶分歧 | **主力特徵候選** |
| `top_acct_lsr`、`global_acct_lsr` 的 z-score | Signal | 單邊擁擠 | 高度相關，需分群去重 |
| `taker_lsvr_1h` 日均、7 日均 | Signal | 主動買賣流 | 用 archive 版本，**不可接 API 版** |
| `rvol_20d` | State | 風險環境 | |
| `btc_ret_7d`、市場平均 funding | State | 市場 regime / shared driver | 也做 DAG 的 confounder |
| `ret_21d`、`volmom_21d` | **Control only** | 已知的動量因子 | ⚠️ 已被先前研究看過，只能當 control 並揭露，不可當 signal 宣稱 |

**Setup v1 草案要點**
- Universe：每日依 `t−1` 為止的 30 日平均成交額排前 100 名，且日成交額 ≥ $5M；含後來下市者。
- Decision：日線 `date` 是 UTC 00:00 開盤時間 → 第 t 日 23:59:59 收盤後才可知；**訊號用到 t 日收盤，在 t+1 日開盤（UTC 00:00）成交**。crypto 24/7 沒有隔夜跳空，但 00:00 剛好是 funding 結算時點，要寫清楚「結算前還是結算後持有」。
- Label：H ∈ {1, 3, 7} 天的 **next-open to next-open 價格報酬 + 期間 funding 現金流**（多頭付、空頭收），與回測成交規則一致。crypto 日曆日就是交易日，buffer 以天計。
- Mapping：每日 z-score 排名，前/後 20% long/short，等權（null）vs inverse-vol（第二 allocator），單檔 ≤ 10%，淨曝險 0，總槓桿 ≤ 1×。
- Cost class：taker 手續費 + half-spread + 平方根衝擊（`engine.py` 的 `CostModel` 可沿用，且已修正過 6 倍高估）+ **funding 現金流**（funding 屬於 label / PnL 的一部分，不是成本參數）。
- Protocol：walk-forward 從 2022-01 開始，訓練窗 12 個月擴張式、步長 1 個月、每月重訓，purge = H 天；holdout 2025-10-25 ~ 2026-10-24，fixed holdout 調參。
- 壓力情境：2022-05 LUNA、2022-11 FTX（都在開發期內）；2025-10 上旬的大規模清算在 holdout 起點前約兩週，要記得看 purge 是否壓到。

**主要風險（"what could break this" 素材）**
- funding 高的幣常常也是剛上市、波動大、流動性快速衰退的幣 → 要做 shared-driver control（rvol、上市天數）。
- 空頭腿可能遭遇軋空，單日 +400% 的事件在資料裡有 938 筆 |ret|>50% → 一定要做 MAE 停損或 vol targeting，並報 with/without。
- METRICS 9.6% 小時缺值（多為上市首日），不可 forward-fill，要 flag & exclude。
- 2021-12 以前的橫截面不足，開發期只有約 3.8 年；每日 IC 的有效樣本要依 H 重疊扣除。

---

### 題目 B（較窄、較快）：大戶 vs 散戶多空比分歧

把題目 A 縮成**單一機制**：`top_pos_lsr`（大戶持倉多空比）與 `global_acct_lsr`（全市場帳戶多空比）的分歧預測未來報酬。

- **Edge**：informational advantage（大戶較有資訊）+ flow predictability。
- **好處**：故事最清楚，DAG 好畫（confounder = 市場報酬、波動；collider 風險 = 用「當日成交額」做條件）；特徵 10 個左右就夠。
- **壞處**：family 太窄，如果 triage 全部 STOP，模型階段會很單薄（雖然負結果流程正確不扣分）。
- **建議**：先以題目 A 為主，B 當 A 裡面的主力特徵子族；W3 若 funding 子族都被 STOP，可以收斂成 B 而不需要換資料。

---

### 題目 C（保守備案）：funding 預測 → carry 策略

Label 改成「未來 7 天的累積 funding」（訊雜比遠高於報酬），模型預測哪些幣會持續付高 funding，再做 delta-neutral carry（做空永續 + 現貨對沖）。

- **好處**：預測問題本身比較好做，IC 會比較漂亮，也不容易被質疑 Sharpe 太高。
- **壞處**：**沒有現貨價格資料**，delta-neutral 的另一腿無法回測，只能用「永續空單 + 市場 beta 對沖」近似；而且「能預測 funding」≠「能賺錢」（funding 高時價格往往也在漲）。改機制要升 v2。
- 適合在 A 的 W3 triage 發現「funding 特徵對 funding 有預測力、但對價格沒有」時，當作第 7 節的 **next step**，不建議當主題。

---

## 4. 不建議的題目

| 題目 | 原因 |
|---|---|
| `volmom_21d` 大幣動量（先前最強候選） | 從 96 個檢定中選出，並且用 2025–2026 OOS 確認過 → 任何最近 12–24 個月的 holdout 都已被污染，**風險是總分上限 60**。除非把它當 non-ML baseline 並在帳本完整揭露 |
| 小市值動量/反轉 | 已在 `SMALLCAP_BACKTEST` 被否決（零成本毛 Sharpe ≈ 0），OOS 也看過了 |
| DEX LP | IL 缺 75%、沒有池內價格可自算 IL、沒有 PIT 宇宙、不是 ML 預測問題，第 4–5 節會很難寫 |
| 只用 `ALPHA/` 小時資料（premium/basis、跨所 funding） | 只有約 4 個月，連 holdout 都不夠 |

---

## 5. 選定題目 A 後，接下來要做的事（按時程）

1. **W1（最優先，在跑任何報酬之前）**：`git init`；寫 `setup.yaml` v1；protocol table；**commit 一份 holdout 封存宣告（日期 + commit hash）**；建立空的 `run_log.csv` 與帳本，把先前研究的 231+ 個 trial 列為 "prior exploration（價格動量 family）"。
2. 確認 `tools/analyze_alpha.py` 看過什麼，決定 holdout 是否需要調整。
3. 把 `tools/` 資料腳本搬進 repo，讓 `data/` 可以從根目錄重建；補 `requirements.txt`（加 `shap`、`optuna`）。
4. W2 之後依 `MIDTERM_REQUIREMENTS.md` 的清單逐週推進。
