# Weekly Rolling Pair Cache v1

用途
- 在週頻估計配對參數（β、z-score），並以日頻回測，嚴格遵守「T+1 收盤成交」時間語義。
- 預先快取 pair 層級的 z 與 β，提升速度與可重現性；同檔保留日頻價（log 與 raw）。

資料來源
- Daily prices: data/prices.pkl（pandas 寬表；index=DatetimeIndex，columns=Symbols；adjusted close）
- Weekly prices: data/weekly_prices.pkl（MultiIndex columns: [Symbol, {W_Close, T1_Close}]；W-FRI 週末）

時間語義（T+1 收盤）
- 於每週最後交易日（WeekEnd）計算 z(t) 與 β(t)。
- 成交在「下一個交易日的收盤」（T+1 close；通常週一，逢假期順延）。
- 回測 PnL 對齊：
  - PnL_t = pos_{t-1} × r_t（也就是使用 pos.shift(1) × r）
  - β 亦採用 beta_{t-1}（即 beta.shift(1)），避免當日使用剛更新的 β。

目錄結構（多組合）

cache/rolling_cache_weekly_v1/
├── manifest.json # 記錄所有已建 L×Z 組合
├── L200/ # formation_length=2.00 年 → L{years×100}
│ ├── Z004/
│ │ ├── meta.json
│ │ └── pairs/
│ │ ├── AMGN__REGN.pkl
│ │ └── ...
│ ├── Z008/
│ │ └── ...
│ └── ...
├── L250/
│ └── Z013/ ...
└── ...


檔案格式
- pairs/{pair_id}.pkl：pandas DataFrame（index=日頻 DatetimeIndex；覆蓋全樣本）
  - z：週頻 z-score，僅在週末日有值，其餘為 NaN（回測仍會 z.shift(1)）
  - beta：日頻有效 β（每個週末估計；自「該週末的 T+1」起 forward-fill 至下一次週末前）
  - px_x, px_y：日收盤對數價（log(adjusted close)）→ 預設供 price_type="log" 使用
  - px_x_raw, px_y_raw：日收盤原始價（adjusted close）→ 供 price_type="raw" 使用
  - week_end_flag：布林；週末日為 True
  - t1_tradeable_flag：布林；在「上個週末的 T+1」當日兩腿皆有日收盤價則 True
- L{...}/Z{...}/meta.json：該組合的建置資訊（formation_length_years、z_window_weeks、built_at 等）
- manifest.json（根）：累積所有已建 L×Z 的摘要（L_tag、Z_tag、built_at、pairs_built 等）

欄位計算規則
- 週資料：W-FRI 週末；W_Close 為該週最後交易日之日收盤；T1_Close 為下一個有效交易日之日收盤。
- OLS：以週頻對數價（log W_Close）做 y=α+βx（固定 Y_on_X）。
  - β_weekly 僅在週末有值；β 日頻有效值將從該週末 T+1 起 forward-fill。
  - 回測計算日度對沖報酬：r_pair = r_y − β_{t-1} · r_x；若 price_type="log"，r=diff(logP)；raw 則 r=pct_change()。
- z-score：以週頻殘差做 rolling 標準化，窗口 z_window_weeks 以「週」為單位。

停損（週數）
- 參數：time_stop_weeks（例：6）
- 回測引擎換算：time_stop_days = ceil(time_stop_weeks × 5)；或採行事曆感知換算（可選）。
- 語義：持倉日數從「成交日」計為 1；達上限當天在 T+1 收盤平倉。

建置工具與指令
- 生成器：scripts/build_weekly_pair_cache.py
- 主要參數：
  - --formation-lengths "2,2.5,3,3.5,4"（年）
  - --z-windows-weeks "4,8,13,26,52"（週）
  - --overwrite-mode overwrite|skip|clean
- 範例（一次建立 5×5 組合）：

python scripts/build_weekly_pair_cache.py ^
--prices data/prices.pkl ^
--weekly data/weekly_prices.pkl ^
--pairs cache/top_pairs_annual.csv ^
--root cache/rolling_cache_weekly_v1 ^
--formation-lengths "2,2.5,3,3.5,4" ^
--z-windows-weeks "4,8,13,26,52" ^
--overwrite-mode overwrite

- 重跑同組合：
  - overwrite：覆寫同名檔案
  - skip：保留既有檔案，不重算
  - clean：先刪該 L/Z 目錄再重建

相容性與 Loader 指南
- RollingCacheLoader 應依 formation_length 與 z_window_weeks 解析至對應路徑 L{L_tag}/Z{zwin}/pairs。
- fields 對應：
  - load_panel(fields=("z",)) → 回傳 [dates × pairs] 的 z 矩陣（z 只有週末有值）
  - load_panel(fields=("beta",)) → 回傳日頻有效 β；回測再 beta.shift(1)
  - load_panel(fields=("px",))：
    - price_type="log" → 使用 px_x/px_y
    - price_type="raw" → 使用 px_x_raw/px_y_raw
- 舊版快取（只有 L{tag}/pairs）：
  - 建議遷移到 L{tag}/Z{default}/pairs（自選預設 Z），或在 Loader 端加 fallback。

邊界與檢核
- 樣本最後一個週末通常無 T+1；該週不可下單（t1_tradeable_flag=False）。
- 若任一腿在 T+1 無日收盤價，該週不可下單。
- OLS 規避零變異：Var(x)=0 → β=NaN；z 在 std 過小時設 NaN。
- Sanity checks（建議）：
  - week_end_flag=True 的日期應等於 weekly_prices 的 index
  - β 在週末當日應為 NaN 或不生效，並自 T+1 起填值
  - 抽樣 pair 繪製 z（週末點）與 beta（階梯狀）與日度 r_pair，檢查時間對齊