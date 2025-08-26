
# walk-forward 圖表 wrapper：

## 針對每個 W（z_window）的最終 L（from wf_final_picks_by_W.csv），產出一張綜合四圖（年度報酬對比、累積淨值、滾動 Sharpe、回撤曲線）且與 _GSPC 買進持有比較

python -m src.wf_plot_picks ^
--prices data\prices.pkl ^
--benchmark-symbol _GSPC ^
--wf-root reports\walkforward_weekly_5bps ^
--out-dir reports\walkforward_figs_5bps

