以下是一份可直接交付給同事/外包團隊的「集成 OOS（top‑M）走動式方法」說明，用於把你週頻版本的流程，等效移植到「日頻」資料上。全程描述方法、介面與檔案，不含程式碼。

集成 OOS（top‑M）— 日頻版方法總覽
- 目的：在每個走動式訓練窗內，於全體候選組合（Set × 參數網格）中選出前 M 名，並在下一個交易期的測試窗以等權或 PSR 權重疊加其日報酬，得到更平滑、更穩健的 OOS 成績。
- 關鍵差異（相對週頻）：
  - 走期與窗長以「日」為基本單位；訓練窗可用過去 N 個交易日/過去 K 個月/過去 1 年等，測試窗常用「下一個曆月」或「下一年」。
  - 訊號/部位每日都可能變化（除非你刻意維持「只在每週五出訊號」的混合週頻邏輯）。
- 核心原則（與你現有回測一致）：
  - 訊號不前視：z_signal = z.shift(1)
  - 成交語義：T+1 close（以 t 的收盤產生訊號，t+1 收盤成交）
  - PnL：pos.shift(1) × r_t
  - β 使用 beta.shift(1)
  - 成本：每腿單邊 cost_bps；翻倉成本自動被 |Δpos| 捕捉

名詞與設定
- Set（組合）：以 Formation_Length × Z_Window 定義
  - Formation_Length（年或日）：形成窗長度；日頻可用年換日（例如 2 年 ≈ 504 交易日）
  - Z_Window（日）：殘差標準化的日滾動窗長度（常見：63、126、252 等）
- 參數 θ（可擴充）：(z_entry, z_exit, time_stop)
  - z_entry ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
  - z_exit ∈ {0.0, 0.5, 1.0}（可支援 crossing-only 出場）
  - time_stop（天）：{none, 20, 30, 60}（或以「週」轉換為天）
- 評分函數（訓練窗）：
  - 以 PSR（Probabilistic Sharpe Ratio，非正態修正）優先，平手看 Sharpe，再看 CumReturn，最後 AnnVol（小者優先）
  - PSR 計算（摘要）：先用未年化 SR = mean/σ，考慮偏度與峰度修正，轉為 z，再取 Φ(z)

走動式（Walk‑Forward）框架（以「月度測試窗」為例）
- 時間切片
  - 訓練窗：前 6 個曆月（或 126 交易日）直到 t−1
  - 測試窗：第 t 個曆月
  - 向前滾動：t = 第一個可測月份 → 最後一個可測月份
  - 備選：年度步階（train：過去 1 年，test：下一年）與你的週頻版完全等價
- 訓練窗內搜尋（當期 t 的前窗）
  1) 對所有 Set = L × Z
  2) 對每個參數格點 θ
  3) 產生「日頻訓練報酬序列」並計算 PSR/Sharpe/CumRet/AnnVol
  4) 以 (PSR, Sharpe, CumRet, −AnnVol) 排序，取前 M 名
  5) 權重：
     - equal：等權 w_i = 1/M
     - psr：w_i ∝ max(0, PSR_i)（負 PSR 設 0 再正規化）
- 測試窗交易（當期 t）
  - 對前 M 名的每一個 (Set, θ) 產生「日頻 OOS 報酬」，依權重線性疊加成組合報酬
  - 計算該期（例如一個月）的 OOS 指標：AnnRet（以日頻年化）、AnnVol、Sharpe、MaxDD、（可選）portfolio_event_days
- 全期聚合
  - 串接所有測試窗的 OOS 日報酬序列，得到全期 OOS 指標與淨值曲線
  - 輸出年度/月份表與全期總結（與基準如 _GSPC 比較；可加 target‑vol 波動對齊）

輸入/輸出（建議實務介面）
- 輸入
  - prices_daily.pkl：日頻 adjusted close，寬表（index=DatetimeIndex、columns=symbol）
  - pairs_daily.csv（或 annual）：包含 stock1、stock2、formation_length、trading_period（若做年度切片）
  - 快取（可選）：每個 pair 的日頻 z、beta、px_x/px_y；或在訓練/測試中當場估（效能較差）
- 設定
  - train_lookback_days 或 train_lookback_months（例如 126d/6m/1y）
  - test_span：monthly（建議）或 yearly
  - grids：z_entry、z_exit、time_stop_days
  - sets：formation_lengths_days 或（年→日換算）× z_windows_days
  - M（ensemble-top）：前幾名；weight_scheme：equal/psr
  - cost_bps、capital、n_pairs_cap
- 輸出
  - oos_by_period.csv（每月/每年的 ensemble OOS 指標；欄位：period, ann_return, ann_vol, sharpe, max_drawdown, selected_topM）
  - oos_returns.csv（全期日報酬與累積淨值）
  - summary.json（全期 OOS 指標）
  - 比較圖：年度柱狀（策略 vs _GSPC）、全期累積（策略 vs _GSPC；x 軸只顯示年份）

日頻版的幾個關鍵細節
- 訊號頻率
  - 完全日頻：每日以 formation 窗估 β、殘差、z，並以日 z 觸發進出。這是最直接的日頻版本。
  - 混合週頻（若想和週版一致）：仍只在每週五（或該週最後交易日）計 z，beta 週更，日內持有；這會降低訊號頻率與翻倉。
- T+1 收盤語義（務必保持）
  - z_signal = z.shift(1)
  - PnL = pos.shift(1) × r_t（日頻 close‑to‑close），成本用 |Δpos| 對應日記入
- 參數網格（日頻建議）
  - z_entry：0.5–3.0（步長 0.5）
  - z_exit：0–1.0（可含 crossing 規則）
  - time_stop_days：none, 10, 20, 30, 60（或依半衰期估個人化上限）
- 排序準則延伸（可選）
  - turnover penalty：score = PSR − λ × turnover_ratio（λ ≈ 0.1–0.3）
  - hysteresis：若新最佳比分數僅小幅領先上一期參數（差 < δ），則沿用上期參數，降低參數跳動
  - FDR 控制：若你在 Set 分群後要做多重比較，可用 BH‑FDR 挑通過者再入榜

與基準比較與波動對齊（target‑vol）
- 為公平對比，建議同時輸出「原始策略」與「波動對齊」策略（static leverage）
  - 計算 realized vol = std(ret_daily) × √252
  - 目標年化波動 σ_target（例如 10%）：λ = σ_target / realized_vol
  - 以 r_scaled = λ × r_daily、equity_scaled = ∏(1 + r_scaled) 繪製曲線與報表
  - 注意這是研究用展示；實務需扣除槓桿成本與保證金約束

效能與工程建議
- 面板重用：對於同一訓練窗，先準備每個 (L, year) 或 (L, 月) 的 YearPanel/MonthPanel，供不同 Z 或 θ 重用，避免反覆 IO 和計算
- 平行化：以 (L, Z) 或 (L, Z, θ) 為粒度；視磁碟 IO 調整 n_jobs（4–8 常見）
- cache：如日頻 universe 大，建議先產 pair 級快取（px_x/px_y、z、beta），以免每次 re‑fit 都重算

日頻部署參數範例（可直接採用）
- 走期：monthly OOS，train_lookback=6m（約 126 交易日），test=next month
- Sets：L ∈ {0.5y, 1.0y, 2.0y} → {126d, 252d, 504d}；Z ∈ {63d, 126d, 252d}
- grids：z_entry ∈ {1.0, 1.5, 2.0, 2.5}；z_exit ∈ {0.0, 0.5}；time_stop ∈ {none, 20, 40}
- M=3；weight=psr；n_jobs=8；cost_bps=5；n_pairs_cap=100
- 輸出：oos_by_month.csv、oos_returns.csv、summary.json、與 _GSPC 的圖（年度可用曆年聚合）

產出物（給 PM/論文的必要圖表）
- 年/月 OOS 柱狀：策略 vs _GSPC
- 全期累積（原始與 target‑vol=10% 的版本）
- 可選：滾動 Sharpe（126d/252d）、回撤時間序列、日報酬分布

驗收清單
- 不前視檢核：z_signal 是否已 shift(1)；PnL 是否用 pos.shift(1)
- 期邊界：每個測試窗僅使用該窗內資料；訓練窗不包含測試窗資料
- 成本：是否按 |Δpos| × (1+|β|) × 名目 × bps 計兩腿成本
- 集成：top‑M 的入選清單、權重與當期 PSR 是否正確記錄到 CSV（selected_topM 與權重）

一句話總結
- 集成 OOS（top‑M）就是：每期用過去一段資料，從所有 Set×參數中挑出最可信的前 M 名，按等權或 PSR 權重把它們在下一期的日報酬疊加起來，並按 T+1 收盤語義與成本規則計算 OOS；月月/年年滾動，如此能有效降低「單一參數挑錯」的風險，提升樣本外穩健度。這套做法與你週頻流程一致，只是時標由週改為日，窗長與 z_window 也改為「日」單位而已。

如果你要，我可以把上面的規格再整理成一頁的技術規格書（含欄位定義表、檔案結構與流程圖），交給工程同事照此實作日頻版本，或直接提供可執行的日頻 wrapper。