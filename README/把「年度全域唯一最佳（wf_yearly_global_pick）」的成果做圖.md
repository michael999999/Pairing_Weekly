# 把「年度全域唯一最佳（wf_yearly_global_pick）」的成果做圖，包含：

年度收益柱狀（策略 vs 基準 _GSPC）
年度 Sharpe 柱狀（策略 vs 基準）
年度 Max Drawdown 柱狀（策略 vs 基準）
全期間累積淨值（策略 vs 基準，x 軸只顯示年份）
說明

讀取 wf_yearly_global_pick 產出的：
global_reopt_oos_yearly.csv（逐年 OOS 指標）
global_reopt_oos_returns.csv（全期 OOS 日報酬與累積淨值）
基準（_GSPC）從 data/prices.pkl 載入；可用參數指定其他代號
x 軸採 YearLocator + '%Y' 只顯示年份，避免擁擠
螢幕會同時印出「全期間策略 vs 基準」的年化指標（Sharpe/AnnRet/AnnVol/MDD）
檔案：src/wf_global_plots.py

使用方式（Windows CMD 範例）

python -m src.wf_yearly_global_pick --out-dir reports\wf_yearly_global_pick ... 先跑出 OOS 結果
python -m src.wf_global_plots ^
--wf-root reports\wf_yearly_global_pick ^
--prices data\prices.pkl ^
--benchmark-symbol _GSPC ^
--out-dir reports\wf_yearly_global_figs