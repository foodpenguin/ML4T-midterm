# ML4T 期中 Proposal 具體要求整理

> 來源：`doc/ML4T_Midterm_Proposal_Guideline.pdf`（Dr. Ting-Yu Chang，2026-10-07，共 50 頁）
> 課本：Machine Learning for Trading, 3rd ed.（以下 Ch. = 課本章節）
> 本文件為逐頁整理後的「可執行清單」，括號內頁碼為 PDF 頁碼。

---

## 0. 一句話總結

期中 = **把 ML4T 工作流程完整走一圈**（Figure 1.2 的每一個方塊都要有），交出「研究文件 PDF + 可重跑的 Git repo + 10 分鐘簡報」。
**評分看流程與誠實，不看 Sharpe 高低**：協定正確但結果普通 ＞ Sharpe 3 但有洩漏。(p.3, p.48)

必須涵蓋：資料基礎建設（PIT 資料、股票池、品質檢查）→ 研究框架 → 特徵工程 → 模型 → 策略設計 → 監控計畫（只需設計，不需實作）。(p.4)
**Evidence boundary**：一開始就封存一段 holdout，最後只碰一次，並與驗證結果分開報告；run log 要顯示在那一次之前跑了幾個 trial。(p.4)

---

## 1. 繳交物（p.44）

| 繳交物 | 格式 | 內容 |
|---|---|---|
| ① Proposal 文件 | PDF，正文 **10–14 頁** + 附錄 | 第 1–8 節（依下方結構）、所有圖附 caption、附錄放研究帳本 |
| ② Repository | Git（GitHub/GitLab），根目錄要有 README | `setup.yaml`、可重建資料的 `data/` 腳本、各階段 notebook（`01_setup` … `07_results`）、`run_log.csv`、`requirements.txt` 或環境檔 |
| ③ 簡報 | **10 分鐘 + 5 分鐘 Q&A，≤ 12 張投影片** | 機制、setup、headline table、6 張必附圖中的 3 張、診斷、下一步 |

- **重現性測試**：老師會 clone repo，從**根目錄**執行資料腳本與 results notebook，把重跑出的 headline table 與**文件中的數字**比對。數字不同不一定扣分，但 README 必須說明原因（隨機種子、資料下載日期、vendor 修正）。
- 簡報第一個問題一定是：**「What was knowable at decision time?」**

---

## 2. 文件必要結構（p.5）

| # | 章節 | 課本章節 | 頁數 |
|---|---|---|---|
| 1 | Hypothesis and sources of edge | Ch. 1, 6 | 1 |
| 2 | Trading setup v1 (`setup.yaml`) and evaluation protocol | Ch. 6 | 1–2 |
| 3 | Data: sources, point-in-time pipeline, quality report | Ch. 2–5 | 1–2 |
| 4 | Labels, features, and triage results | Ch. 7–10 | 2 |
| 5 | Models, validation, interpretation, uncertainty | Ch. 11–15 | 2 |
| 6 | Backtest protocol, baseline, portfolio, costs, risk | Ch. 16–19 | 2–3 |
| 7 | Results, diagnosis, and next iteration | Ch. 20 | 1–2 |
| 8 | Monitoring and deployment plan | Ch. 25–26 | 0.5–1 |
| A | Appendix: research ledger, run log, repository map | Ch. 6, 16 | 不限 |

- 正文目標 **10–14 頁**。
- **每一節結尾都要有一段「What could break this」**（什麼情況會讓本節結論失效）。交件前十問中答「否／不確定」的項目，要在對應節的這段揭露。(p.5, p.47)

---

## 3. 各節具體要求

### 第 1 節　Hypothesis and sources of edge（Ch. 1, 6）(p.8)

- 進入點：**建議 Mechanism-first**（先有經濟理由，限縮搜尋空間、預先知道什麼會壞）。
  - Prediction-first 也可以，但必須有嚴格統計預算：固定搜尋集合、walk-forward、多重檢定控制；選這條路第 4 節會更長。
- 兩種進入點都要通過共同防線（也是評分重點）：decision-time correctness、time-aware validation、multiple-testing control、realistic backtests、drift monitoring。
- **必寫三件事**：
  1. 用**兩句話**說明機制（為何此效應存在？對手是誰？為何對方願意接受？）
  2. 優勢來源（Table 6.1）：risk compensation / liquidity provision / funding constraints / flow predictability / informational advantage
  3. **證偽條件**（什麼結果會讓你停止）

### 第 2 節　Trading setup v1 + Evaluation protocol（Ch. 6）(p.9–11)

**(a) `setup.yaml`（任何回測前寫好、放進 repo）**

| 欄位 | 要寫什麼 | 課本例（ETF momentum） |
|---|---|---|
| Universe and eligibility | PIT 規則、流動性篩選 | 100 檔跨資產 ETF；trailing ADV ≥ $10M |
| Decision schedule | snapshot 時點、執行時點、延遲 | 月底收盤；隔日開盤執行 |
| Admissible information | 回看期、各資料的可得延遲 | 價格到 t；基本面延遲 60 天 |
| Score-to-trade mapping | 進場、出場、部位大小、**明確的 no-trade 狀態** | 排名取前 10，等權，只做多 |
| Constraints and risk controls | 槓桿、集中度、曝險上限 | 滿倉、不加槓桿、單檔 ≤ 10% |
| Cost-model class | 適用哪些摩擦成本（類型現在固定，參數可之後估） | $0.0035/股 + 分級 half-spread |
| Version | 機制改變時升版 | `setup_version: v1` |

- 版本規則：**調參數（如門檻 1.8→2.0）留在 v1**；**改機制（long-only→long/short、加入資金成本等新成本項）= v2**，結果不可直接比較。

**(b) Evaluation protocol（看結果之前就承諾）**

- **六項設計承諾**：訓練窗長度、步長、重訓頻率、label horizon、最長特徵回看期、測試期。
- **禁止 random k-fold**，用 walk-forward。
- Buffer：**label buffer = label horizon（purging）**，例如 label 看 21 天，每個驗證區塊前的最後 21 個交易日要從訓練集移除；若訓練資料可能落在驗證之後，再加 feature buffer（embargo）。**一律以交易日計算**。
- **Holdout：最後 12–24 個月**，在提交 setup 的當天封存；並說明調參方式是 nested walk-forward 或 fixed holdout。
- 交付：第 2 節放 **protocol table**，**實際切分日期寫進 run log**（找不到切分日期 → 結果只算 exploration，不算 evidence）。

**(c) Search accounting / 研究帳本（第一次回測前就要存在，計分）** (p.11)

> 帳本本身放在**附錄 A**；第 2 節要做的是承諾記帳規則並事先登記 confirmatory test。

- 四層分類：**Strategy → Trial family → Trial → Run**。
- 每個 run 記錄：provenance（git commit、資料快照）、configuration、評估區間、指標、decision 與通過/未通過的 gate。CSV 或 Markdown 表即可（MLflow 可選）。
- **事先登記 confirmatory test**：哪個指標、哪個門檻、哪個 holdout。
- Trial 數會用在第 7 節的 **Deflated Sharpe Ratio**。
- 修 bug 不算 trial；**看到結果不好而改 lookback 算 trial**。

### 第 3 節　Data（Ch. 2–5）(p.15–17)

**(a) 資料路線（標準相同）**

| 路線 | 範例 | 必須記錄 |
|---|---|---|
| A. 課本 case study | `etfs`（日頻、免費、免 API key，建議）、`us_equities_panel`、`cme_futures`、`fx_pairs`、`crypto_perps_funding` | 從哪個 `setup.yaml` 起步；對 universe 或頻率的任何更動 |
| B. 自備資料（鼓勵，但說明負擔較重） | yfinance、TWSE/TPEx 日資料、FRED、**Binance public API**、券商匯出 | vendor、下載日期、timestamp 語意、調整方法、identifier 穩定性、revision policy |
| 兩者皆可 | 基本面、文字、另類資料作為 add-on | 每筆紀錄的 availability timestamp；必須說得出 first-observable time |

- 建模前鎖定四件事（Ch. 2）：**timestamp semantics、corporate-action adjustment、identifier stability、revision handling**。**說不出某值何時可知，就不能用。**

**(b) PIT 正確性三層次** (p.16)

| 層次 | 要求 |
|---|---|
| Value | 每個特徵只用 `available_at ≤ decision time` 的紀錄；非價格資料要附 **lag distribution**（zero-lag 紀錄是紅旗，可仿 Figure 2.1） |
| Universe | 資格用 t 時點資訊計算；**要包含下市/剔除的標的**，否則說明理由與偏誤方向 |
| Identifier | 代號變更與重複使用以生效日處理（台股注意代號重新分配與股別變更） |

- 交付：**PIT checklist 表**（values / universe / identifiers × pass / fail / N/A，仿 Table 2.2）。Fail 可接受，但要揭露並討論偏誤方向。

**(c) Data quality report（一頁即可）** (p.17)

必附表/圖：
- 每期 eligible 標的數（coverage over time）
- 缺值與處理方式（**禁止默默 forward-fill**，要 flag & exclude）
- Corporate-action 檢查：至少一檔有分割或配息的標的，raw vs adjusted（仿 Figure 2.3）
- OHLC invariant 違反數與重複筆數
- Survivorship：多少標的離開樣本、何時

必寫敘述：
- 儲存格式與**如何精確重建資料**（腳本名稱、Parquet 分區）
- Session 與時區慣例（台股 13:30 收盤；crypto 24/7；FX 紐約 17:00）
- 若由 tick 建 bar：取樣規則與 bar 數（Ch. 3）
- 若用合成資料（Ch. 5）：**只能做穩健性檢查，不能當主要結果**

### 第 4 節　Labels, features, triage（Ch. 7–10）(p.20–23)

**(a) Label**
- 必寫：label 類型（固定期間報酬 / 門檻分類 / triple barrier）、horizon、**對應的執行價格慣例**。
- **Label 必須與回測成交規則一致**：回測隔日開盤成交，label 就要從隔日開盤算起（close-to-close vs next-open 在日頻股票每筆可差 50–100 bps）。
- **回報重疊**：名目樣本數 vs **有效樣本數**，所有信賴區間都用有效樣本（課本例：248,436 → 約 11,830）。
- **預設用連續報酬 label**；用分類要說明理由。

**(b) Feature 表（每個特徵一列，仿 Table 8.1）**

欄位：Feature｜Role（Signal / State）｜Driver hypothesis（機制）｜Lookback / cadence（須 ≤ label horizon）｜Representation（level/change/rank/z-score、是否 vol-scaled）｜Known failure mode

課本範例列（可當模板）：

| Feature | Role | Driver hypothesis | Lookback / cadence | Representation | Known failure mode |
|---|---|---|---|---|---|
| mom_126d | Signal | Persistence（underreaction） | 126d, monthly | Vol-scaled, cross-sectional rank | Regime concentration, crashes |
| rvol_20d | State | Risk environment | 20d, daily | Percentile vs own history | Single shock dominates |
| carry_roll | Signal | Risk compensation | Front vs deferred, daily | Annualized % | Wrong front contract |
| corr_spy_tlt_63d | State | Hedge regime | 63d, daily | Level | Nonsynchronous trading |

- 三步篩選（Ch. 8）：horizon alignment、driver hypothesis、role separation。**說不出機制 = 只是資料型態，不是特徵。**
- 預算：**10–30 個候選特徵**，用相關性分群去重。

**(c) Triage report：五道關卡（Ch. 7）**

| Gate | 內容 |
|---|---|
| 1 Correctness | t 時可觀察、對齊 label、無洩漏；**timing placebo**（往回移 5–21 天，IC 應消失） |
| 2 Association | 分 fold IC + **HAC CI**、ICIR、同號 fold 比例、**至少 3 個 horizon 的 IC decay**（只給 pooled IC 不夠） |
| 3 Shape | 五分位報酬是否單調（或 U 型）；分類則 ROC / PR |
| 4 Feasibility | 特徵隱含換手；gross spread 須 **≥ 2 × 假設來回成本** |
| 5 Search accounting | 搜尋集合大小；**Benjamini–Hochberg 或 Holm–Bonferroni，α = 0.05**；報 effect size |

- 交付：**每個候選**（含失敗者）都標 **PROCEED / REVISE / STOP**；多數是 REVISE 很正常，列出失敗者是評分的一部分。

**(d) Mechanism plausibility（前 3 名特徵）**
- 畫 **DAG**：confounders（要控制）、mediators（不要控制）、colliders（絕不控制）。
- **四個簡易檢查**：timing placebo、shared-driver control（如 market beta、sector）、regime heterogeneity（VIX 或 realized-vol 三分位下符號是否一致）、event-time alignment。
- 文字/另類資料（Ch. 10）：availability timestamp、publication lag、去重、embedding 模型 training cutoff。
- **加分（選做）**：對部署特徵做 Double ML + block-permutation refutation（Ch. 15），算入 rubric 的 causal credibility。

### 第 5 節　Models and validation（Ch. 11–15）(p.26–27)

**Baseline ladder**

| 階 | 模型 | 必/選 | 規範 |
|---|---|---|---|
| 1 | 非 ML 規則 baseline | **必** | 由機制推出（如 top-N momentum、門檻規則）（Ch. 16） |
| 2 | Ridge / Elastic Net | **必** | α 掃 **≥ 6 個數量級**；standardize 與 winsorize **只在訓練 fold 上 fit** |
| 3 | LightGBM | **必** | 淺樹、**MAE loss**、early stopping、Optuna **最多 50–100 trials** |
| 4 | 深度序列 / latent factor | 選 | 須有時間結構或橫斷面因子的假說；須以 **paired HAC delta** 勝過最佳表格模型（Ch. 13–14） |
| 5 | Ensemble | 選 | Top-3 z-score 平均；報告 dispersion 降低，不只 peak Sharpe（Ch. 20） |

- **共同尺度**：驗證 fold 上的**平均每日 Spearman IC + Newey–West 95% CI**；CI 不含 0 才算有訊號。

**Validation**
- Walk-forward fold **完全照第 2 節承諾**；fold 日期放附錄。
- 超參數用 nested CV 或 fixed holdout，**絕不在 sealed holdout 上調**。
- **每 fold IC 圖 + HAC band**（仿 Figure 11.8），不只平均。
- **Paired comparison table**：每一階 vs 前一階，ΔIC 與 CI。

**Interpretation & uncertainty**
- 部署模型的 **SHAP summary**：符號一致性、量級合理性、跨 fold 穩定性。
- 至少討論 **1 個「出乎意料」的 attribution**，對照機制。
- **Conformal prediction interval 80–90%**，每 fold 實際覆蓋率；**覆蓋率 < 80% 要標示**。
- 明確區分 descriptive（SHAP）vs causal（Ch. 15）。
- 驗證與 holdout 不一致時，先診斷漂移類型：prediction-quality drift / portfolio-translation drift / regime change（Ch. 20），**不要直接再調超參數**。

### 第 6 節　Backtest, portfolio, costs, risk（Ch. 16–19）(p.30–34)

順序：固定協定 → 非 ML baseline → 真實成本 → 風控 → **最後才看權益曲線**。

**(a) 回測協定六要素（仿 Table 16.2）**

| 要素 | 必須說明 | 不確定時的預設 |
|---|---|---|
| Signal timing | 訊號時點、成交時點、延遲 | 收盤訊號 → 隔日開盤成交；**同 bar 成交 = lookahead** |
| Rebalance & holding | 頻率、持有期、出場 | 月或週；持有期 = label horizon |
| Position sizing | 規則與上限 | Top-N 等權，單檔 ≤ 10% |
| Fill assumptions | 價格基準、部分成交 | Open 或 VWAP；缺 bar 不成交（**不可在 forward-filled bar 上交易**） |
| Constraints | 槓桿、放空、現金 | Long-only、滿倉；**在模擬內檢查**（事後檢查 = 沒模擬） |
| Costs | 模型類型與參數 | 依標的的經驗法則（Table 18.2）+ 敏感度掃描 |

- 引擎：權重型、路徑無關規則可用 vectorized；**若有停損、回撤斷路器、現金限制影響下一步，需 event-driven 引擎**（ml4t-backtest、backtrader、Zipline）。**註明用哪個及原因。**

**(b) Baseline 與成本敏感度** (p.31)
- 必：非 ML baseline + **被動 benchmark**（60/40、等權或指數），**完全相同協定**。課本 baseline Sharpe 0.65 輸給 60/40 的 0.80，刻意不美化；**baseline 看起來太好，要懷疑有洩漏**。
- 必：**成本掃描 0 → ≥ 4 × 假設 per-leg 成本**，報 **break-even cost c\* = gross excess return / turnover**（是診斷上限，不是預測）。
- 第 6 節要有一張類似 **Figure 16.1** 的圖：net Sharpe 與 net CAGR 對 per-leg cost，並標出假設成本（課本 ETF baseline 的 break-even 約 166 bp）。
- 報：年化雙邊換手、平均持有期、成本區間（hard floor / gray zone / comfortable）。
- 台股：賣出證交稅 0.3%、手續費 0.1425%；選擇權成本依權利金比例。

**(c) Portfolio construction** (p.32)
- 必：**等權（null）+ 至少一種**（inverse vol、score weight 或 HRP），同訊號、同共變異窗口。
- 報：Sharpe、MDD、turnover、集中度（HHI 或 effective bets）、best–worst spread。
- Mean-variance 只能搭配 **Ledoit–Wolf shrinkage + 限制條件**，並先寫 allocator term sheet（目標、輸入、限制、再平衡、評估計畫）。注意 Markowitz curse：課本無限制的範例最後只剩兩檔資產。
- 預期：配置是二階效果（課本中位數 spread 0.49 Sharpe），**救不了弱訊號**。

**(d) 最低風險 term sheet** (p.33)

| 控制 | 必要內容 | 課本 |
|---|---|---|
| Risk control matrix | market、factor、leverage、concentration、liquidity、model、operational：各給 proxy、control、檢查頻率 | Table 19.2 |
| Tail & path risk | 歷史 95% VaR / CVaR（regime-conditional）；MDD 深度、持續時間、恢復時間 | Ch. 19.3–19.4 |
| Exposure decomposition | 對 benchmark 的 rolling beta；有因子模型則附 factor loadings + HAC band | Ch. 19.5 |
| Stress test | **兩段歷史事件**（如 2020 COVID、2022 升息）+ **成本加壓（spread × 3）** | Tables 19.5–19.6 |
| Adaptive control | **一個**只用 t−1 資訊的無洩漏 overlay：vol targeting，或用 MAE/MFE 分布校準的停損 | Ch. 19.7 |
| Kill-switch ladder | 回撤門檻 → 動作 → 負責人，如 5% watch、10% 減半、15% halt、20% unwind | Table 19.7 |

- **Overlays refine, never redeem**：有/無 overlay 都要報；若靠 overlay 才轉正，要直說。

### 第 7 節　Results, diagnosis, next iteration（Ch. 20）(p.34, p.38–40)

**(0) Statistical inference：Sharpe 是估計值** (p.34)
> PDF 把這頁放在第 6 部分，但講稿明說「your Section 7 must treat it that way」，評分也歸在 Results, inference（10 分），所以寫在**文件第 7 節**。
- **Sharpe 95% CI**：SE(SR) ≈ √((1 + SR²/2)/T)；驗證與 holdout 都要報（12 個月日資料 SE ≈ ±1，所以 holdout Sharpe 1.0 幾乎無法和 0 區分，要直說）。
- **Deflated Sharpe Ratio（或 PSR）**：用帳本 trial 數 + skew、kurtosis（100 個零技能策略最佳 Sharpe 可達 2.54）。
- **Minimum track record length**：Y_min ≈ ((z_α + z_β)/SR)²，α = 0.05、power 0.80 下需要幾年（SR 0.5 約 25 年）。
- 範例：「SR = 1.2 [−0.3, 2.7]，40 trials」比單寫「SR = 1.2」更有說服力。

**(a) Headline table（結果節開頭，仿 Table 20.2）**

| Stage | Metric | Validation | Holdout | Benchmark |
|---|---|---|---|---|
| Signal | Daily IC [HAC 95% CI]、ICIR、正 IC fold 比例 | … | … | — |
| Portfolio | Gross Sharpe [CI]、MDD、turnover | … | … | EW / 60-40 / index |
| Costs | 假設成本下 Net Sharpe；break-even cost（bp/leg） | … | … | — |
| Risk overlay | 加 overlay 後 Net Sharpe 與 MDD | … | … | — |
| Inference | Deflated Sharpe（trials = N）、minimum track record | … | … | — |

表後**一句話**依 Ch. 20 五種型態歸類：**hold / decay or sign flip / strengthen / never resolved / marginal behind a cost screen**。

**(b) 六張必附圖**（caption 寫問題、內文寫一句話答案）
1. **Equity curves**：策略、非 ML baseline、被動 benchmark（皆為淨成本），**標出 validation/holdout 分界線**
2. **Rolling 3-month IC**（或 rolling Sharpe）含零線（仿 Figure 11.8）
3. **Gross-to-net Sharpe bar** 並標 break-even cost（仿 Figure 18.4）
4. **Regime slice**：volatility × trend 下的 Sharpe 與 drawdown，門檻 PIT 計算（仿 Table 16.6）；**不可試多種 regime 定義只挑好看的（regime snooping）**
5. **SHAP beeswarm**（部署模型），對照機制：首要特徵符號是否如預期
6. **Drawdown chart** + kill-switch 門檻水平線

**(c) Diagnosis & next step**
- 在哪一關失去價值：signal / portfolio / cost / holdout gate？
- 驗證與 holdout 不一致 → 指名漂移類型並附證據：prediction quality（holdout IC 下降）/ portfolio translation（IC 穩但 Sharpe 下降）/ regime change（看 regime slice）。
- 指出 binding constraint：liquidity、regime persistence、cost cadence、label/feature richness、construction robustness。
- **只寫「一個」具體下一步**（如「日頻改週頻再測 cost gate」）；**「再多調參數」不算**。

### 第 8 節　Monitoring & deployment plan（Ch. 25–26）(p.41)

半頁左右，**只需設計、不需實作**：
- **Parity plan**：同一份程式如何上線；逐階段比對 raw data → features → predictions → signals → orders（Ch. 25）
- **Monitoring**：30/60/90 天 rolling Sharpe、drawdown、hit rate、IC；live vs backtest 的 realization ratio
- **Drift**：前 5 特徵的 **PSI**（0.10–0.25 觀察、> 0.25 重訓）+ SHAP-rank 檢查
- **Lifecycle rules**：明確的 retrain / pause / retire 觸發條件；**投入資金前先 shadow mode**
- 區分 **bug vs 統計衰退 vs 成本上升**，各對應不同處置：修 bug / 重訓 / 改頻率（Ch. 26）

### 附錄 A　Research ledger, run log, repository map
- 研究帳本（Strategy → Trial family → Trial → Run）、run log、repo 結構說明。
- 原文對位置的規定：**切分日期寫在 run log**（p.10）；**fold 日期放附錄**（p.27）；run log 要看得出碰 holdout 之前跑了幾個 trial（p.4）。

---

## 4. 評分（100 分）(p.45)

| 項目 | 分數 | 滿分樣貌 |
|---|---|---|
| Hypothesis, setup v1, protocol commitment | 15 | 機制明確；`setup.yaml` 完整；協定與 holdout 在看結果前固定；帳本存在 |
| Data and PIT correctness | 15 | 三層 PIT checklist；品質報告；資料可由腳本重建 |
| Labels, features, triage | 15 | Label 對齊執行；特徵表含機制；套用關卡；列出失敗者；多重檢定校正 |
| Models and validation | 15 | 完成 baseline ladder；paired comparison + HAC CI；SHAP 與 conformal |
| Backtest, costs, portfolio, risk | **20** | 協定固定；非 ML baseline 與 benchmark；成本掃描與 break-even；風險 term sheet；overlay 有/無 |
| Results, inference, diagnosis | 10 | Headline table；validation vs holdout；DSR 含 trial 數；歸類型態；一個下一步 |
| Reproducibility and presentation | 10 | Repo 可重跑；圖清楚；Q&A 誠實 |

**罰則**
- **任何 lookahead 或 holdout 污染 → 總分上限 60**。
- 流程正確的負結果或無訊號結果**不扣分**；**無法解釋的 Sharpe > 3 會扣分**（「寧可看到誠實的 0.4」）。

**等第範例** (p.48)
- **A**：ETF momentum × 波動狀態閘門，免費資料；holdout Sharpe 0.6 [−0.4, 1.6]，18 個 logged trials；break-even 45 bp；regime slice 顯示失效處；指名下一步；全部可重跑。
- **B**：機制好、資料乾淨，但 label 用 close-to-close 而成交在 next-open；配置比較成為主要結果。
- **上限 60**：用今日成分股建的台股小型股 panel 得 Sharpe 3.4、全樣本 fit scaler、調參後重跑 holdout、沒有帳本。

---

## 5. 時程（六週六個檢查點；W1–W5 週五交，W6 為最終繳交）(p.46)

| 週 | 主題 | 檢查點內容 |
|---|---|---|
| W1 | Setup & ledger | 機制、`setup.yaml` v1、protocol table、**封存 holdout**、**commit 空帳本**（在下載快速看一眼以外的資料之前完成） |
| W2 | Data | 資料可重建、PIT checklist、品質報告草稿 |
| W3 | Labels, features, triage | Label spec、特徵表、triage 表（PROCEED/REVISE/STOP） |
| W4 | Models | 規則 baseline、Ridge、LightGBM（驗證 fold）、paired comparison、SHAP |
| W5 | Backtest, costs, risk | 協定、baseline vs benchmark、成本掃描、風險 term sheet、overlay |
| W6 | Holdout & write-up | **只跑一次 holdout**、headline table、診斷、三項繳交物 |

檢查點很短（半頁或一個 notebook），目的是確保 W6 時 holdout 仍封存、帳本是真的而非事後補。

---

## 6. 交件前十問（p.47）

若任一題答「否／不確定」：修正，或在該節「What could break this」揭露。

- [ ] 1. 每個特徵都能說出何時變得可知？
- [ ] 2. Universe 用 t 時點資訊建立，且包含離開者？
- [ ] 3. Label 的價格慣例與回測成交規則一致？
- [ ] 4. Scaler、winsorization 界限、門檻都只在訓練 fold 上 fit？
- [ ] 5. Label 與 feature buffer 以交易日計算，且每個 fold 都套用？
- [ ] 6. Holdout 在規格凍結後只碰過一次？
- [ ] 7. 帳本記錄每個 adaptive choice，DSR 使用該次數？
- [ ] 8. 成本依標的收取（稅、手續費、價差），並有掃描與 break-even？
- [ ] 9. 限制與停損在模擬內以 t−1 資訊執行？
- [ ] 10. 他人能從 repo 根目錄重建資料與 headline table？

---

## 7. Repo 交付物對照清單

> 以下目錄結構是**建議**。原文只列出必須包含的內容，並要求從根目錄執行；資料夾怎麼擺沒有規定。

```
<repo-root>/
├── README.md               # 如何重跑；數字差異原因（seed、下載日期、vendor revision）
├── setup.yaml              # setup_version: v1（含 universe、schedule、info lag、mapping、constraints、cost class）
├── requirements.txt        # 或 environment.yml
├── run_log.csv             # 每個 run：git commit、資料快照、config、評估區間、指標、decision/gate；切分日期
├── data/                   # 重建資料集的腳本（建議 partitioned Parquet）
└── notebooks/
    ├── 01_setup.ipynb
    ├── …                   # 每個 pipeline 階段一本
    └── 07_results.ipynb    # 產出 headline table
```

> PDF 原文只舉例 `01_setup … 07_results`，中間的命名（02_data、03_features …）由各組自訂，與課本 case study 一致即可。
