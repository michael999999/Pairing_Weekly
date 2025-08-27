# 周線配對交易（Weekly Pairs Trading）全量流程 README

本 README 以「全量資料」為例，完整說明從原始日價到週頻快取、單次與批次回測、參數搜尋（Grid Search）、走動式（Walk-Forward）與圖表輸出的端到端流程與指令。所有方法皆遵守「T+1 收盤成交」語義，並以日頻報酬計算 PnL。

關鍵要點
- 價格來源：日頻 adjusted close（data/prices.pkl；寬表，DatetimeIndex × Symbols）
- 週頻語義：週末（週五或該週最後交易日）出訊號；於下一交易日的收盤（T+1 close）成交
- 不前視回測：
  - 部位以前一日訊號決策：z_signal = z.shift(1)
  - PnL 使用前一日部位：pos.shift(1) × r_t
  - β 於週末估，從 T+1 起 forward-fill；回測使用 beta.shift(1)
- 成本：每腿單邊（per-leg, one-way）cost_bps；預設 5 bps，可配置
- 方向性配對：pair_id = stock1__stock2，A__B 與 B__A 視為不同配對
- 程式風格：註解繁體中文；print/log 英文

---

## 0) 需求與環境

- Python 3.9+（建議）
- 需要套件：pandas、numpy、joblib、matplotlib、seaborn
- 專案結構（建議）
  ```
  data/
    prices.pkl                # 日頻 adjusted close（寬表）
  cache/
    top_pairs.csv             # 原始半年度配對名單（YYYYH1/HH2）
    top_pairs_annual.csv      # 轉年度後的名單（YYYY）
    rolling_cache_weekly_v1/  # 週頻 pair-cache（Lxxx/Zyyy）
  reports/
  src/
    __init__.py               # 必須存在，方便 python -m 啟動
    backtest_full.py
    batch_weekly.py
    grid_search_weekly.py
    plot_thesis_figs.py
    walkforward_weekly.py
    wf_plot_picks.py
    wf_yearly_reopt.py
    wf_yearly_global_pick.py
    wf_global_plots.py
  scripts/
    build_weekly_prices.py
    convert_pairs_to_annual.py
    build_weekly_pair_cache.py
    period_coverage_heatmap.py
  ```

執行方式
- 請在專案根目錄執行
- 以套件模式啟動：python -m src.module_name
- Windows CMD 換行使用 ^；PowerShell 使用 `；Linux/macOS 使用 \

---

## 1) 由日價產生週頻基準價（W_Close / T1_Close）

輸入：data/prices.pkl  
輸出：data/weekly_prices.pkl（MultiIndex columns: [Symbol, {W_Close, T1_Close}]）

Windows
```
python scripts\build_weekly_prices.py ^
  --input data\prices.pkl ^
  --output data\weekly_prices.pkl ^
  --freq W-FRI
```
Linux/macOS
```
python scripts/build_weekly_prices.py \
  --input data/prices.pkl \
  --output data/weekly_prices.pkl \
  --freq W-FRI
```

說明
- W_Close：該週最後交易日之日收盤
- T1_Close：該週週末之後「下一個有效交易日」的收盤（遇假期順延）
- 樣本最後一週通常沒有 T1_Close（NaN），該週不可交易

---

## 2) 半年度配對名單轉年度（H1→全年，刪除 H2）

輸入：cache/top_pairs.csv  
輸出：cache/top_pairs_annual.csv（trading_period 改為 YYYY；trading_start=YYYY-01-01；trading_end=YYYY-12-31；依 formation_length、trading_period 排序）
```
python scripts\convert_pairs_to_annual.py ^
  --input cache\top_pairs.csv ^
  --output cache\top_pairs_annual.csv
```

---

## 3) 建立週頻 pair-cache（多組 Formation_Length × Z-Window）

輸入：data/prices.pkl、data/weekly_prices.pkl、cache/top_pairs_annual.csv  
輸出：cache/rolling_cache_weekly_v1/L{L}/Z{Z}/pairs/{pair_id}.pkl

（全量建立 5×5：L=2,2.5,3,3.5,4；Z=4,8,13,26,52）
```
python scripts\build_weekly_pair_cache.py ^
  --prices data\prices.pkl ^
  --weekly data\weekly_prices.pkl ^
  --pairs cache\top_pairs_annual.csv ^
  --root cache\rolling_cache_weekly_v1 ^
  --formation-lengths "2,2.5,3,3.5,4" ^
  --z-windows-weeks "4,8,13,26,52" ^
  --overwrite-mode overwrite
```

每個 pair 檔包含
- z：僅週末有值（週 t 的 W_Close），其餘為 NaN
- beta：日頻有效值；在週末估，從 T+1 起 forward-fill 至下一週末前
- px_x/px_y：日對數價（log adjusted close）；另保留 px_x_raw/px_y_raw
- week_end_flag、t1_tradeable_flag（可用於更嚴格的成交檢核）

---

## 4) 單次回測（Smoke Test）

以 L=2.0 / Z=13 / 年度 2019 為例：
```
python -m src.backtest_full ^
  --top-csv cache\top_pairs_annual.csv ^
  --cache-root cache\rolling_cache_weekly_v1 ^
  --price-type log ^
  --formation-lengths "2" ^
  --trading-periods 2019 ^
  --z-window 13 ^
  --time-stop-weeks 6 ^
  --ignore-selection-formation ^
  --out-dir reports\weekly_smoke_L2_Z13
```

檢核重點
- z 只在週末有值
- β 呈階梯狀，從週末後 T+1 才生效（回測再用 beta.shift(1)）
- 成交日（T+1）當天 PnL 為 0，隔日才開始計入（pos.shift(1) × r）

---

## 5) 批次回測（多 Z-Window）

一次跑 Z ∈ {4,8,13,26,52}；每次 run 內可同時含多個 L。
```
python -m src.batch_weekly ^
  --top-csv cache\top_pairs_annual.csv ^
  --cache-root cache\rolling_cache_weekly_v1 ^
  --price-type log ^
  --formation-lengths "2,2.5,3,3.5,4" ^
  --trading-periods all ^
  --z-windows "4,8,13,26,52" ^
  --time-stop-weeks 6 ^
  --ignore-selection-formation ^
  --out-root reports\weekly_batch
```

輸出
- reports/weekly_batch/Z004 等子資料夾（各 run）
- reports/weekly_batch/_summary/combined_metrics.csv、pivot_sharpe.csv/png、pivot_mdd.csv/png

---

## 6) 參數搜尋（Grid Search，Set = L × Z）

在每個 Set（固定 L、Z）內搜尋 θ = (z_entry, z_exit, time_stop_weeks)，以 Sharpe（全期）優先，平手看總報酬，再看年化波動；逐筆交易層級計算 Profit Factor、勝率、平均持有天數。支援平行化（--n-jobs, --backend）。

全量範例
```
python -m src.grid_search_weekly ^
  --top-csv cache\top_pairs_annual.csv ^
  --cache-root cache\rolling_cache_weekly_v1 ^
  --price-type log ^
  --formation-lengths all ^
  --z-windows all ^
  --trading-periods all ^
  --grid-z-entry "0.5,1.0,1.5,2.0,2.5" ^
  --grid-z-exit "0.0,0.5" ^
  --grid-time-stop "none,6" ^
  --cost-bps 5 ^
  --capital 1000000 ^
  --n-pairs-cap 60 ^
  --ignore-selection-formation ^
  --n-jobs 8 ^
  --backend loky ^
  --out-dir reports\gridsearch_weekly
```

輸出
- reports/gridsearch_weekly/Lxxx_Zyyy/：best_params.json、equity_curve_full.csv、yearly_metrics.csv
- reports/gridsearch_weekly/_summary/best_sets.csv（25 行最佳 Set 清單）
- 螢幕表格欄位（對齊）：formation_length, z_window, z_entry, z_exit, time_stop, cost_bps, cum_return, ann_return, ann_vol, sharpe, max_drawdown, total_trades（pair 級總筆數）, win_rate, avg_duration_days, profit_factor

注意
- total_trades 為「配對層級 round-trip 總數」；可能大於年度週數（合理）
- 需要「投組事件日」可另行擴充（非預設欄位）

---

## 7) 論文圖表（由 Grid Search 輸出）

從 best_sets.csv 與各 set 的檔案產出圖（含基準 _GSPC）。
```
python -m src.plot_thesis_figs ^
  --grid-root reports\gridsearch_weekly ^
  --prices data\prices.pkl ^
  --benchmark-symbol _GSPC ^
  --out-dir reports\thesis_figs
```

產生
- L×Z 熱力圖（Sharpe/MDD；CSV+PNG）
- 針對 Sharpe 最佳的 Set：
  - 年度收益柱狀（策略 vs _GSPC）
  - 年度 Sharpe 柱狀
  - 年度超額報酬（策略 − _GSPC）
  - 全期累積淨值（Strategies vs _GSPC；x 軸僅顯示年份）
  - 滾動 Sharpe（252d；可增 126d）
  - 日報酬分布、回撤時間序列

---

## 8) 走動式（Walk-Forward，Set 層 CV）

以年度排序「先定參、後評估」：
- 訓練窗：前 n 年（--train-periods），使用年度 Leave-One-Out（LOO）交叉驗證，在訓練窗內為每個 Set 選 θ*
- 測試窗：僅在下一年用 θ* 交易，得到 OOS
- 聚合：計算全期 OOS Sharpe、PSR（非正態修正）、p-value；對每個 W 用 BH-FDR 控制 FDR 選「最終 L」

全量範例（train-periods=3 → OOS 只會有 2018、2019）
```
python -m src.walkforward_weekly ^
  --top-csv cache\top_pairs_annual.csv ^
  --cache-root cache\rolling_cache_weekly_v1 ^
  --price-type log ^
  --formation-lengths all ^
  --z-windows all ^
  --trading-periods all ^
  --train-periods 3 ^
  --grid-z-entry "0.5,1.0,1.5,2.0,2.5" ^
  --grid-z-exit "0.0,0.5" ^
  --grid-time-stop "none,6" ^
  --cost-bps 5 ^
  --capital 1000000 ^
  --n-pairs-cap 60 ^
  --ignore-selection-formation ^
  --n-jobs 8 ^
  --backend loky ^
  --fdr-q 0.1 ^
  --out-dir reports\walkforward_weekly
```

圖表（每個 W 的最終 L 綜合圖，年度/累積/滾動 Sharpe/回撤 vs _GSPC）
```
python -m src.wf_plot_picks ^
  --wf-root reports\walkforward_weekly ^
  --prices data\prices.pkl ^
  --benchmark-symbol _GSPC ^
  --out-dir reports\walkforward_figs
```

提示
- 想要 2016–2019 都有 OOS，請把 --train-periods 設為 1（或增加更早年度）

---

## 9) 年度走動式（Yearly Re-Optimization，Set 各自最佳）

「每個 Set（L×Z）各自」在訓練窗內挑唯一最佳 θ*，用來做下一年 OOS，逐年列印。  
（這個模式會輸出很多組 Set 的年度結果；適合比較不同 Set 的穩健性。）

範例
```
python -m src.wf_yearly_reopt ^
  --top-csv cache\top_pairs_annual.csv ^
  --cache-root cache\rolling_cache_weekly_v1 ^
  --price-type log ^
  --formation-lengths all ^
  --z-windows all ^
  --trading-periods all ^
  --train-periods 1 ^
  --grid-z-entry "0.5,1.0,1.5,2.0,2.5,3.0" ^
  --grid-z-exit "0.0,0.5,1.0,1.5,2.0,2.5" ^
  --grid-time-stop "none,6,9" ^
  --cost-bps 5 ^
  --capital 1000000 ^
  --n-pairs-cap 60 ^
  --ignore-selection-formation ^
  --n-jobs 8 ^
  --backend loky ^
  --out-dir reports\wf_yearly_reopt
```

---

## 10) 年度走動式（Yearly Global Pick，平行化，全域唯一最佳）

你若要「每年只選出 1 組全域唯一最佳（L×Z×θ）」來做 OOS，請使用本模式：
- 訓練窗（前 n 年）掃「所有 L×Z × 參數格點」，平行化搜尋每個 (L,Z) 的最佳 θ，然後在所有 (L,Z) 之中挑出唯一本期最佳組合 (L*, Z*, θ*)
- 測試窗：僅用 (L*, Z*, θ*) 做下一年 OOS
- 逐年滾動，只輸出每年一行；最後輸出全期 OOS 累積指標

範例（train-periods=1 → OOS=2016–2019 全部）
```
python -m src.wf_yearly_global_pick ^
  --top-csv cache\top_pairs_annual.csv ^
  --cache-root cache\rolling_cache_weekly_v1 ^
  --price-type log ^
  --formation-lengths all ^
  --z-windows all ^
  --trading-periods all ^
  --train-periods 1 ^
  --grid-z-entry "0.5,1.0,1.5,2.0,2.5,3.0" ^
  --grid-z-exit "0.0,0.5,1.0,1.5,2.0,2.5" ^
  --grid-time-stop "none,6,9" ^
  --cost-bps 5 ^
  --capital 1000000 ^
  --n-pairs-cap 60 ^
  --ignore-selection-formation ^
  --n-jobs 8 ^
  --backend loky ^
  --out-dir reports\wf_yearly_global_pick
```

圖表（年度收益/Sharpe/MDD 與全期累積 vs _GSPC）
```
python -m src.wf_global_plots ^
  --wf-root reports\wf_yearly_global_pick ^
  --prices data\prices.pkl ^
  --benchmark-symbol _GSPC ^
  --out-dir reports\wf_yearly_global_figs
```

---

## 11) 覆蓋度檢核（年度 × formation_length）

快速檢視名單涵蓋度（每年 × L 的 pair 數量）：
```
python scripts\period_coverage_heatmap.py ^
  --pairs cache\top_pairs_annual.csv ^
  --out-csv reports\coverage_counts.csv ^
  --out-png reports\coverage_heatmap.png
```

---

## 參數對照與語義

- formation_length（L，年）：形成窗長度（例 2.5）
- z_window（Z，週）：殘差標準化的週滾動窗（例 52）
- z_entry / z_exit：進出場 z 門檻
- time_stop_weeks：時間停損（週）；回測換算為 ceil(weeks × 5) 個交易日
- price_type：log（diff 計報酬）或 raw（pct_change）
- cost_bps：每腿單邊成本（bps）
- total_trades（gridsearch/walk-forward 輸出）：配對層級 round-trip 總數（pair-level）；不是投組事件數
- T+1 收盤成交語義：
  - 週末（W_Close）產生 z
  - T+1 收盤建倉/平倉
  - PnL 使用 pos.shift(1) × r_t；beta 使用 beta.shift(1)

---

## 常見問題與排錯

- 模組匯入錯誤（attempted relative import）：請確認 src 下有 __init__.py，並用 python -m src.module 方式啟動
- 找不到 L/Z 快取：檢查 cache/rolling_cache_weekly_v1/Lxxx/Zyyy/pairs 是否已有對應檔案
- OOS 年度過少：--train-periods 越大，最早 OOS 就越晚；例：5 年資料 + train=3 → 只有最後 2 年 OOS
- total_trades 看似過大：它是 pair 級總筆數，相加會大於週數上限；如需投組事件日，請加自訂統計
- x 軸標籤擁擠：圖表已用 YearLocator 只顯示年份；如需更改，請調整 wrapper 的日期格式器

---

## 端到端「全量」範例流程（建議順序）

1) 週價檔  
```
python scripts\build_weekly_prices.py --input data\prices.pkl --output data\weekly_prices.pkl
```
2) 年度化名單  
```
python scripts\convert_pairs_to_annual.py --input cache\top_pairs.csv --output cache\top_pairs_annual.csv
```
3) 週頻快取（全 L×Z）  
```
python scripts\build_weekly_pair_cache.py --prices data\prices.pkl --weekly data\weekly_prices.pkl --pairs cache\top_pairs_annual.csv --root cache\rolling_cache_weekly_v1 --formation-lengths "2,2.5,3,3.5,4" --z-windows-weeks "4,8,13,26,52" --overwrite-mode overwrite
```
4) 單次回測（驗證）  
```
python -m src.backtest_full --top-csv cache\top_pairs_annual.csv --cache-root cache\rolling_cache_weekly_v1 --price-type log --formation-lengths "2" --trading-periods 2019 --z-window 13 --time-stop-weeks 6 --ignore-selection-formation --out-dir reports\weekly_smoke_L2_Z13
```
5) 批次回測（多 Z）  
```
python -m src.batch_weekly --top-csv cache\top_pairs_annual.csv --cache-root cache\rolling_cache_weekly_v1 --price-type log --formation-lengths "2,2.5,3,3.5,4" --trading-periods all --z-windows "4,8,13,26,52" --time-stop-weeks 6 --ignore-selection-formation --out-root reports\weekly_batch
```
6) 參數搜尋（Set 內網格）  
```
python -m src.grid_search_weekly --top-csv cache\top_pairs_annual.csv --cache-root cache\rolling_cache_weekly_v1 --price-type log --formation-lengths all --z-windows all --trading-periods all --grid-z-entry "0.5,1.0,1.5,2.0,2.5" --grid-z-exit "0.0,0.5" --grid-time-stop "none,6" --cost-bps 5 --capital 1000000 --n-pairs-cap 60 --ignore-selection-formation --n-jobs 8 --backend loky --out-dir reports\gridsearch_weekly
```
7) 論文圖表（GridSearch）  
```
python -m src.plot_thesis_figs --grid-root reports\gridsearch_weekly --prices data\prices.pkl --benchmark-symbol _GSPC --out-dir reports\thesis_figs
```
8) 走動式（Set 層 CV）  
```
python -m src.walkforward_weekly --top-csv cache\top_pairs_annual.csv --cache-root cache\rolling_cache_weekly_v1 --price-type log --formation-lengths all --z-windows all --trading-periods all --train-periods 3 --grid-z-entry "0.5,1.0,1.5,2.0,2.5" --grid-z-exit "0.0,0.5" --grid-time-stop "none,6" --cost-bps 5 --capital 1000000 --n-pairs-cap 60 --ignore-selection-formation --n-jobs 8 --backend loky --fdr-q 0.1 --out-dir reports\walkforward_weekly
```
9) 走動式 Picks 圖  
```
python -m src.wf_plot_picks --wf-root reports\walkforward_weekly --prices data\prices.pkl --benchmark-symbol _GSPC --out-dir reports\walkforward_figs
```
10) 年度走動式（各 Set 各自最佳；選用）  
```
python -m src.wf_yearly_reopt --top-csv cache\top_pairs_annual.csv --cache-root cache\rolling_cache_weekly_v1 --price-type log --formation-lengths all --z-windows all --trading-periods all --train-periods 1 --grid-z-entry "0.5,1.0,1.5,2.0,2.5,3.0" --grid-z-exit "0.0,0.5,1.0,1.5,2.0,2.5" --grid-time-stop "none,6,9" --cost-bps 5 --capital 1000000 --n-pairs-cap 60 --ignore-selection-formation --n-jobs 8 --backend loky --out-dir reports\wf_yearly_reopt
```
11) 年度走動式（全域唯一最佳；平行化）  
```
python -m src.wf_yearly_global_pick --top-csv cache\top_pairs_annual.csv --cache-root cache\rolling_cache_weekly_v1 --price-type log --formation-lengths all --z-windows all --trading-periods all --train-periods 1 --grid-z-entry "0.5,1.0,1.5,2.0,2.5,3.0" --grid-z-exit "0.0,0.5,1.0,1.5,2.0,2.5" --grid-time-stop "none,6,9" --cost-bps 5 --capital 1000000 --n-pairs-cap 60 --ignore-selection-formation --n-jobs 8 --backend loky --out-dir reports\wf_yearly_global_pick
```
12) 年度全域最佳 圖表（策略 vs _GSPC）  
```
python -m src.wf_global_plots --wf-root reports\wf_yearly_global_pick --prices data\prices.pkl --benchmark-symbol _GSPC --out-dir reports\wf_yearly_global_figs
```

---
