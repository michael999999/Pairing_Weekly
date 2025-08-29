
# compare_daily_weekly_sets.py

- 讀取「日頻」與「週頻」的 sets summary（CSV），在同成本假設下，依 formation_length 與
  「日↔週等效 z-window 映射」成對比較。
- 僅用 summary 近似比較（未使用時間序列），提供：
  1) 風險對齊（target volatility）下的年化報酬與 MDD 近似（ret@TV, MDD@TV）
  2) Sharpe 直接比較（已風險標準化）
  3) 對照表 matched_comparison.csv（完整指標）
  4) 成對長條圖：Sharpe、ret@TV、MDD@TV
  5) 散佈圖：Sharpe、ret@TV
  6) 勝率熱圖：按 formation_length 與按 family（D↔W）
  7) 漂亮排版：
     - paired_table_pretty_2lines.txt（兩行一組，Daily/Weekly 對齊；Weekly 行附 Δ）
     - summary.txt（表格風格：AvgΔ, Wins, Loses, Ties）
- 注意：此為 summary 近似版；論文最終檢定請用時間序列重算與統計檢定（NW/JK-M/Bootstrap）


===================================================================================================================

成本 5（主結果；日頻用 gateN）

python -m src.compare_daily_weekly_sets ^
--daily-csv reports\summary\daily_sets_summary.csv ^
--weekly-csv reports\summary\weekly_sets_summary.csv ^
--daily-scenario-id price_log__cost5__gateN ^
--weekly-cost-bps 5 ^
--target-vol 0.10 ^
--mapping "21:4,42:8,63:13,126:26,252:52" ^
--select-rule sharpe ^
--out-dir reports\summary\compare_daily_weekly_cost5_gateN

===================================================================================================================

成本 0（附錄）

python -m src.compare_daily_weekly_sets ^
--daily-csv reports\summary\daily_sets_summary.csv ^
--weekly-csv reports\summary\weekly_sets_summary.csv ^
--daily-scenario-id price_log__cost0__gateN ^
--weekly-cost-bps 0 ^
--target-vol 0.10 ^
--mapping "21:4,42:8,63:13,126:26,252:52" ^
--select-rule sharpe ^
--out-dir reports\summary\compare_daily_weekly_cost0_gateN