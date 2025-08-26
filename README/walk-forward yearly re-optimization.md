
#「典型年度走動式（walk-forward yearly re-optimization）」腳本：

訓練窗：以前 n 年（--train-periods）組成的年度集合
參數選擇：在訓練窗內以「唯一最佳」θ* = (z_entry, z_exit, time_stop_weeks)（評分以 Sharpe → 累積報酬 → 年化波動）
測試窗：只在下一年用 θ* 交易，取得該年的 OOS 指標
逐年重複，最後輸出「2015–2019 每年的最佳參數與績效」與「全期間累計績效」
螢幕輸出右對齊、年份不帶小數；time_stop 也靠右
產出 CSV/JSON（每年記錄與全期摘要）

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