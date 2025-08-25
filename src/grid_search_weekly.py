#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid Search & Summary for Weekly Pairs Backtest
- 以 Set = formation_length × z_window（週窗）為單位做參數搜尋與彙總
- 參數網格：z_entry × z_exit × time_stop_weeks（含 none）
- 評分準則：Sharpe（全期）優先；平手看總報酬（較大者），再看年化波動（較小者）
- 產出：
  1) 每個 Set 的最佳參數與績效（螢幕表格 + best_sets.csv）
  2) 年度與全期結果（equity_curve_full.csv、yearly_metrics.csv）
  3) 無交易期數統計（Σ|Δpos|=0）
- 平行化：joblib（loky/threading），建議 loky

註解：繁體中文；print/log 為英文
"""

import argparse
import json
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# 匯入週頻快取 Loader
try:
    from .cache_loader import RollingCacheLoader
except Exception:
    from src.cache_loader import RollingCacheLoader


# ====== 小工具 ======

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_floats_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_ints_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_time_stop_grid(s: str) -> List[Optional[int]]:
    """
    將字串解析為時間停損（週數）清單；支援 'none' 表示無時間停損。
    回傳單位為「週」（整數）或 None；之後再換算為天。
    """
    out: List[Optional[int]] = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if t in ("none", "nan", ""):
            out.append(None)
        else:
            out.append(int(float(t)))
    return out

def ann_metrics(returns: pd.Series, freq: int = 252) -> Dict[str, float]:
    """年化指標（日頻年化）。"""
    r = returns.dropna()
    if len(r) == 0:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0)
    mu = r.mean() * freq
    vol = r.std(ddof=1) * sqrt(freq) if r.std(ddof=1) > 0 else 0.0
    sharpe = mu / vol if vol > 0 else 0.0
    return dict(ann_return=float(mu), ann_vol=float(vol), sharpe=float(sharpe))

def max_drawdown_curve(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0

def to_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    """保證 pair_id 欄位存在（方向化）。"""
    if "pair_id" in df.columns:
        return df
    if "stock1" in df.columns and "stock2" in df.columns:
        df["pair_id"] = df["stock1"].astype(str) + "__" + df["stock2"].astype(str)
        return df
    if "pair" in df.columns:
        df["pair_id"] = df["pair"].astype(str)
        return df
    raise KeyError("Selection CSV needs 'pair_id' or ('stock1','stock2') columns.")

def list_LZ_from_cache(root: Path) -> Tuple[List[float], List[int]]:
    """從快取根目錄掃描可用的 L 與 Z。"""
    Ls = []
    Zs = set()
    if not root.exists():
        return Ls, sorted(list(Zs))
    for Ldir in root.iterdir():
        if Ldir.is_dir() and Ldir.name.upper().startswith("L"):
            try:
                Lval = float(int(Ldir.name[1:])) / 100.0
            except Exception:
                continue
            Ls.append(Lval)
            for Zdir in Ldir.iterdir():
                if Zdir.is_dir() and Zdir.name.upper().startswith("Z"):
                    try:
                        Zval = int(Zdir.name[1:])
                        Zs.add(Zval)
                    except Exception:
                        continue
    return sorted(Ls), sorted(list(Zs))


# ====== 部位生成 ======

def build_positions(z: pd.DataFrame, z_entry: float, z_exit: float, time_stop_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    由 z 生成部位與持有日數（外部應先 shift 保證不前視）。
    - 規則：|z| >= z_entry 進場；|z| <= z_exit 出場；持有日數達 time_stop_days 亦出場
    - 回傳：pos, days（與 z 同形狀；pos ∈ {-1,0,+1}）
    """
    pos = z.copy() * np.nan
    days = z.copy() * 0.0
    for pid in z.columns:
        z_ser = z[pid]
        pos_ser = pd.Series(index=z.index, dtype="float64")
        dcount = pd.Series(index=z.index, dtype="float64")
        last_pos = 0.0
        hold_days = 0
        for i, _ in enumerate(z.index):
            sig = z_ser.iloc[i]
            curr = last_pos
            if curr == 0:
                if pd.notna(sig) and sig >= z_entry:
                    curr = -1.0
                    hold_days = 1
                elif pd.notna(sig) and sig <= -z_entry:
                    curr = +1.0
                    hold_days = 1
                else:
                    curr = 0.0
                    hold_days = 0
            else:
                exit_flag = (pd.notna(sig) and abs(sig) <= z_exit) or (hold_days >= time_stop_days)
                if exit_flag:
                    curr = 0.0
                    hold_days = 0
                else:
                    hold_days += 1
            pos_ser.iloc[i] = curr
            dcount.iloc[i] = hold_days
            last_pos = curr
        pos[pid] = pos_ser
        days[pid] = dcount
    return pos, days


# ====== 評估核心（單年度） ======

@dataclass
class YearPanel:
    dates: pd.DatetimeIndex
    z_signal: pd.DataFrame   # [dates × pairs]
    r_pair: pd.DataFrame     # [dates × pairs]
    beta_abs: pd.DataFrame   # [dates × pairs] 用於成本
    pair_ids: List[str]
    w_per_pair: float        # 等權權重（每對資金分配）

def prepare_year_panel(loader: RollingCacheLoader,
                       pair_ids: List[str],
                       tp_start: str,
                       tp_end: str,
                       price_type: str) -> Optional[YearPanel]:
    """讀入該年度所需矩陣，產出 YearPanel。"""
    date_range = (tp_start, tp_end)
    panel_z = loader.load_panel(pair_ids, fields=("z",), date_range=date_range, join="outer", allow_missing=True)
    panel_b = loader.load_panel(pair_ids, fields=("beta",), date_range=date_range, join="outer", allow_missing=True)
    panel_px = loader.load_panel(pair_ids, fields=("px",), date_range=date_range, join="outer", allow_missing=True)
    if panel_z.empty or panel_b.empty or panel_px.empty:
        return None

    # 對齊日期
    dates = panel_z.index.union(panel_b.index).union(panel_px.index).sort_values()
    z = panel_z.reindex(dates)["z"]
    beta = panel_b.reindex(dates)["beta"]
    # 價格欄位
    cols0 = panel_px.columns.get_level_values(0)
    if "px_x" in cols0 and "px_y" in cols0:
        xkey, ykey = "px_x", "px_y"
    elif "px_x_raw" in cols0 and "px_y_raw" in cols0:
        xkey, ykey = "px_x_raw", "px_y_raw"
    else:
        return None
    px = panel_px.reindex(dates)
    if price_type == "log":
        rx = px[xkey].diff()
        ry = px[ykey].diff()
    else:
        rx = px[xkey].pct_change()
        ry = px[ykey].pct_change()

    beta_lag = beta.shift(1)
    r_pair = (ry - beta_lag * rx)

    # 不前視：z 使用 t-1 決策
    z_signal = z.shift(1)

    # 每年度等權配置
    n_pairs = len(z.columns)
    if n_pairs == 0:
        return None
    w = 1.0 / float(n_pairs)

    return YearPanel(
        dates=dates,
        z_signal=z_signal,
        r_pair=r_pair,
        beta_abs=beta_lag.abs().fillna(0.0),
        pair_ids=list(z.columns),
        w_per_pair=w
    )


def eval_year(year_panel: YearPanel,
              z_entry: float,
              z_exit: float,
              time_stop_days: int,
              cost_bps: float,
              capital: float) -> Tuple[pd.Series, Dict[str, float], Dict[str, float], bool]:
    """
    單年度評估：回傳
    - ret_series（日報酬，已含成本）
    - metrics：年化、MDD 等
    - trade_stats：總交易數、勝率、平均持有天數、profit factor
    - no_trade_flag：Σ|Δpos|==0
    """
    zsig = year_panel.z_signal
    r_pair = year_panel.r_pair
    beta_abs = year_panel.beta_abs
    w = year_panel.w_per_pair
    cap = float(capital)

    # 無時間停損以非常大的天數表徵
    tstop = time_stop_days if time_stop_days is not None else 10**9

    # 生成部位（不前視的 z_signal 已由外部提供）
    pos, days = build_positions(zsig, z_entry=z_entry, z_exit=z_exit, time_stop_days=tstop)

    # T+1 收盤成交：PnL 使用前一日部位
    pnl_ex = (pos.shift(1) * r_pair * (w * cap)).sum(axis=1)

    # 成本：|Δpos| × 名目（Y 腿 + |β|×X 腿）
    dpos = pos.fillna(0.0).diff().abs()
    traded_notional = ((w * cap) * (1.0 + beta_abs) * dpos).sum(axis=1)
    cost = traded_notional * (float(cost_bps) / 10000.0)

    pnl_net = pnl_ex - cost
    ret = pnl_net / cap
    equity = (1.0 + ret.fillna(0.0)).cumprod()
    dd = max_drawdown_curve(equity)

    m = ann_metrics(ret)
    m["max_drawdown"] = float(dd.min()) if len(dd) else 0.0

    # 交易統計（以 pair 層聚合為 portfolio 統計近似）
    total_trades = 0
    wins = 0
    losses = 0
    sum_win = 0.0
    sum_loss = 0.0
    durations: List[int] = []

    cost_rate = float(cost_bps) / 10000.0
    for pid in pos.columns:
        pos_s = pos[pid]
        r_s = r_pair[pid]
        beta_abs_s = beta_abs[pid]

        # 單對的日 PnL（含成本）
        pnl_ex_s = (pos_s.shift(1) * r_s * (w * cap))
        cost_s = ((w * cap) * (1.0 + beta_abs_s) * pos_s.fillna(0.0).diff().abs()) * cost_rate
        pnl_net_s = pnl_ex_s - cost_s

        holding = pos_s.fillna(0.0) != 0.0
        # 找出連續持倉區段
        if holding.any():
            # 利用差分找區段起訖
            h = holding.astype(int).values
            idx = np.where(np.diff(np.r_[0, h, 0]) != 0)[0]
            # idx 兩兩成對：[start0, end0, start1, end1, ...]（end 為第一個 False 的位置）
            for j in range(0, len(idx), 2):
                start = idx[j]           # 第一個 True 的位置
                end = idx[j+1] - 1       # 最後一個 True 的位置
                # 交易日 PnL 累積（pos.shift(1) → 從 start+1 到 end）
                if end >= start + 1:
                    # 取對應日期切片
                    dates = pos_s.index
                    pnl_slice = pnl_net_s.iloc[start+1 : end+1]  # end 包含
                    pnl_trade = float(pnl_slice.sum())
                    dur_days = int(end - start + 1)
                    total_trades += 1
                    durations.append(dur_days)
                    if pnl_trade > 0:
                        wins += 1
                        sum_win += pnl_trade
                    elif pnl_trade < 0:
                        losses += 1
                        sum_loss += -pnl_trade  # 累加損失的絕對值
                    else:
                        # 平手不計入勝/負
                        pass

    win_rate = (wins / total_trades) if total_trades > 0 else np.nan
    profit_factor = (sum_win / sum_loss) if sum_loss > 0 else (np.inf if sum_win > 0 else np.nan)
    avg_duration_days = (np.mean(durations) if durations else np.nan)

    trade_stats = dict(
        total_trades=int(total_trades),
        win_rate=float(win_rate) if win_rate == win_rate else np.nan,
        avg_duration_days=float(avg_duration_days) if avg_duration_days == avg_duration_days else np.nan,
        profit_factor=float(profit_factor) if profit_factor not in (np.inf, -np.inf) else np.inf
    )

    # 無交易年度：Σ|Δpos|==0
    no_trade_flag = bool(dpos.sum(axis=1).sum() == 0.0)

    return ret, m, trade_stats, no_trade_flag


# ====== Set 內參數網格搜尋 ======

@dataclass
class BestResult:
    L: float
    Z: int
    z_entry: float
    z_exit: float
    time_stop_weeks: Optional[int]   # None 表示無時間停損
    cost_bps: float
    cum_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    avg_duration_days: float
    profit_factor: float
    no_trade_years: int
    years_covered: List[str]
    equity_full: pd.Series
    yearly_rows: List[Dict[str, object]]

def select_best(results: List[Dict]) -> Dict:
    """依準則挑選最佳：Sharpe → Total Return → Volatility（小）。"""
    # 準備排序鍵（Sharpe 降冪、Total Return 降冪、AnnVol 升冪）
    def key(r):
        return (
            round(r["sharpe"], 10),
            round(r["cum_return"], 10),
            -round(r["ann_vol"], 10)  # 之後我們用 reverse=True
        )
    # 排序（reverse=True 使 Sharpe/Return 最大者在前；AnnVol 經符號反轉等同於最小者在前）
    results_sorted = sorted(results, key=key, reverse=True)
    return results_sorted[0]


def run_set(L: float,
            Z: int,
            sel_df: pd.DataFrame,
            loader_root: str,
            price_type: str,
            trading_periods: List[str],
            z_entry_grid: List[float],
            z_exit_grid: List[float],
            tstop_grid_weeks: List[Optional[int]],
            cost_bps: float,
            capital: float,
            n_pairs_cap: int,
            ignore_selection_formation: bool,
            log_prefix: str = "") -> Optional[BestResult]:
    """執行單一 Set（L×Z）的參數搜尋，回傳最佳結果。"""
    # 初始化 Loader
    loader = RollingCacheLoader(
        root=loader_root,
        price_type=price_type,
        formation_length=float(L),
        z_window=int(Z),
        log_level="ERROR"
    )

    # 為每個年度準備 YearPanel
    year_panels: Dict[str, YearPanel] = {}
    years_used: List[str] = []

    for tp in trading_periods:
        # 篩選年度名單（是否忽略 selection 的 formation_length）
        if ignore_selection_formation:
            g = sel_df[sel_df["trading_period"].astype(str) == tp].copy()
        else:
            g = sel_df[(sel_df["trading_period"].astype(str) == tp) &
                       (pd.to_numeric(sel_df["formation_length"], errors="coerce") == float(L))].copy()
        if g.empty:
            continue
        # Top-K by rank_final
        if "rank_final" in g.columns:
            g = g.sort_values(["rank_final"], ascending=True)
        pair_ids = g["pair_id"].dropna().astype(str).unique().tolist()[:int(n_pairs_cap)]
        if not pair_ids:
            continue

        # 交易窗（全年）
        t_start = str(g["trading_start"].iloc[0]) if "trading_start" in g.columns else f"{tp}-01-01"
        t_end   = str(g["trading_end"].iloc[0])   if "trading_end" in g.columns   else f"{tp}-12-31"

        panel = prepare_year_panel(loader, pair_ids, t_start, t_end, price_type=price_type)
        if panel is None:
            continue
        year_panels[tp] = panel
        years_used.append(tp)

    if not year_panels:
        return None

    # 掃描參數網格
    grid_results: List[Dict] = []
    for ze in z_entry_grid:
        for zx in z_exit_grid:
            for tsw in tstop_grid_weeks:
                time_stop_days = None if tsw is None else int(ceil(float(tsw) * 5.0))

                # 累積全期日報酬與交易統計
                all_returns = []
                total_trades = 0
                wins = 0
                sum_dur = 0.0
                n_trades_for_avg = 0
                sum_win = 0.0
                sum_loss = 0.0
                no_trade_years = 0

                yearly_rows: List[Dict[str, object]] = []

                for y in years_used:
                    panel = year_panels[y]
                    ret_y, m_y, tstats_y, no_trade = eval_year(
                        year_panel=panel,
                        z_entry=float(ze),
                        z_exit=float(zx),
                        time_stop_days=time_stop_days,
                        cost_bps=float(cost_bps),
                        capital=float(capital)
                    )
                    # 年度累加
                    all_returns.append(ret_y)
                    total_trades += int(tstats_y["total_trades"])
                    if not np.isnan(tstats_y["win_rate"]):
                        wins += int(round(tstats_y["win_rate"] * tstats_y["total_trades"]))
                    if not np.isnan(tstats_y["avg_duration_days"]):
                        sum_dur += tstats_y["avg_duration_days"] * tstats_y["total_trades"]
                        n_trades_for_avg += tstats_y["total_trades"]
                    # profit factor 部分：我們用年度 trade-level 累積的總利潤/損失近似（eval_year 已計算）
                    # 但我們只拿到 pf 數值，為避免誤加，改以總 trades/勝率/平均時長近似，pf 於全期重新估算不便於沒有細項。
                    # 故這裡先留待全期計算（簡化：用年度 ret 難以還原）。我們在 eval_year 已經用 pair 級別逐筆 PnL 計算，這裡難整併。
                    if no_trade:
                        no_trade_years += 1

                    # 年度記錄
                    yearly_rows.append(dict(
                        formation_length=L,
                        z_window=Z,
                        trading_period=y,
                        ann_return=float(m_y["ann_return"]),
                        ann_vol=float(m_y["ann_vol"]),
                        sharpe=float(m_y["sharpe"]),
                        max_drawdown=float(m_y["max_drawdown"]),
                        total_trades=int(tstats_y["total_trades"]),
                        win_rate=float(tstats_y["win_rate"]) if tstats_y["win_rate"] == tstats_y["win_rate"] else np.nan,
                        avg_duration_days=float(tstats_y["avg_duration_days"]) if tstats_y["avg_duration_days"] == tstats_y["avg_duration_days"] else np.nan
                    ))

                # 全期合併
                if not all_returns:
                    continue
                ret_full = pd.concat(all_returns).sort_index()
                eq_full = (1.0 + ret_full.fillna(0.0)).cumprod()
                dd_full = max_drawdown_curve(eq_full)
                m_full = ann_metrics(ret_full)
                m_full["max_drawdown"] = float(dd_full.min()) if len(dd_full) else 0.0
                cum_return = float(eq_full.iloc[-1] - 1.0) if len(eq_full) else 0.0

                # 全期勝率（以年度勝數近似）
                win_rate_full = (wins / total_trades) if total_trades > 0 else np.nan
                avg_dur_full = (sum_dur / n_trades_for_avg) if n_trades_for_avg > 0 else np.nan

                grid_results.append(dict(
                    L=L, Z=Z,
                    z_entry=float(ze), z_exit=float(zx),
                    time_stop_weeks=(int(tsw) if tsw is not None else None),
                    cost_bps=float(cost_bps),
                    cum_return=float(cum_return),
                    ann_return=float(m_full["ann_return"]),
                    ann_vol=float(m_full["ann_vol"]),
                    sharpe=float(m_full["sharpe"]),
                    max_drawdown=float(m_full["max_drawdown"]),
                    total_trades=int(total_trades),
                    win_rate=float(win_rate_full) if win_rate_full == win_rate_full else np.nan,
                    avg_duration_days=float(avg_dur_full) if avg_dur_full == avg_dur_full else np.nan,
                    profit_factor=np.nan,  # 全期 PF 嚴謹需彙整逐筆交易 PnL；此處可留空或後續延伸
                    no_trade_years=int(no_trade_years),
                    years_covered=list(years_used),
                    equity_full=eq_full,
                    yearly_rows=yearly_rows
                ))

    if not grid_results:
        return None

    best = select_best(grid_results)

    return BestResult(
        L=best["L"],
        Z=best["Z"],
        z_entry=best["z_entry"],
        z_exit=best["z_exit"],
        time_stop_weeks=best["time_stop_weeks"],
        cost_bps=best["cost_bps"],
        cum_return=best["cum_return"],
        ann_return=best["ann_return"],
        ann_vol=best["ann_vol"],
        sharpe=best["sharpe"],
        max_drawdown=best["max_drawdown"],
        total_trades=best["total_trades"],
        win_rate=best["win_rate"],
        avg_duration_days=best["avg_duration_days"],
        profit_factor=best["profit_factor"],
        no_trade_years=best["no_trade_years"],
        years_covered=best["years_covered"],
        equity_full=best["equity_full"],
        yearly_rows=best["yearly_rows"]
    )


# ====== 主流程 ======

def print_table(best_list: List[BestResult]):
    """美化列印結果表。"""
    headers = ["formation_length","z_window","z_entry","z_exit","time_stop","cost_bps",
               "cum_return","ann_return","ann_vol","sharpe","max_drawdown",
               "total_trades","win_rate","avg_duration_days","profit_factor"]
    widths = [18, 9, 8, 8, 10, 9, 12, 12, 10, 8, 13, 13, 9, 18, 13]
    # 表頭
    line = " ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    # 內容
    for r in sorted(best_list, key=lambda x: (x.L, x.Z)):
        def pct(x): return f"{x:.2%}" if x == x and np.isfinite(x) else "NA"
        def num(x, n=2): return f"{x:.{n}f}" if x == x and np.isfinite(x) else "NA"
        row = [
            num(r.L, 1).rjust(widths[0]-1).rjust(widths[0]),   # 對齊
            str(r.Z).rjust(widths[1]),
            num(r.z_entry, 1).rjust(widths[2]),
            num(r.z_exit, 1).rjust(widths[3]),
            (f"{r.time_stop_weeks}w" if r.time_stop_weeks is not None else "none").rjust(widths[4]),
            num(r.cost_bps, 1).rjust(widths[5]),
            pct(r.cum_return).rjust(widths[6]),
            pct(r.ann_return).rjust(widths[7]),
            pct(r.ann_vol).rjust(widths[8]),
            num(r.sharpe, 2).rjust(widths[9]),
            pct(r.max_drawdown).rjust(widths[10]),
            str(r.total_trades).rjust(widths[11]),
            pct(r.win_rate).rjust(widths[12]),
            num(r.avg_duration_days, 1).rjust(widths[13]),
            (num(r.profit_factor, 2) if np.isfinite(r.profit_factor) else "inf").rjust(widths[14]),
        ]
        print(" ".join(row))


def main():
    ap = argparse.ArgumentParser(description="Grid-search weekly pairs backtest per Set (L×Z) with aggregation.")
    ap.add_argument("--top-csv", type=str, default="cache/top_pairs_annual.csv", help="Selection CSV.")
    ap.add_argument("--cache-root", type=str, default="cache/rolling_cache_weekly_v1", help="Weekly cache root.")
    ap.add_argument("--price-type", type=str, default="log", choices=["log","raw"], help="Price type.")
    ap.add_argument("--formation-lengths", type=str, default="all", help='Comma floats in years or "all".')
    ap.add_argument("--z-windows", type=str, default="all", help='Comma ints in weeks or "all".')
    ap.add_argument("--trading-periods", type=str, default="all", help='Comma years or "all".')
    ap.add_argument("--grid-z-entry", type=str, default="0.5,1.0,1.5,2.0,2.5", help="Grid for z_entry.")
    ap.add_argument("--grid-z-exit", type=str, default="0.0,0.5", help="Grid for z_exit.")
    ap.add_argument("--grid-time-stop", type=str, default="none,6", help='Grid for time stop (weeks); supports "none".')
    ap.add_argument("--cost-bps", type=float, default=5.0, help="Per-leg, one-way cost in bps.")
    ap.add_argument("--capital", type=float, default=1_000_000.0, help="Capital for scaling.")
    ap.add_argument("--n-pairs-cap", type=int, default=60, help="Max pairs per year.")
    ap.add_argument("--ignore-selection-formation", action="store_true", help="Reuse same pair list across L.")
    ap.add_argument("--n-jobs", type=int, default=8, help="Parallel jobs.")
    ap.add_argument("--backend", type=str, default="loky", choices=["loky","threading"], help="Joblib backend.")
    ap.add_argument("--out-dir", type=str, default="reports/gridsearch_weekly", help="Output root.")

    args = ap.parse_args()

    out_root = Path(args.out_dir)
    summary_dir = out_root / "_summary"
    ensure_dir(summary_dir)

    # 讀 selection
    print(f"[INFO] Loading selection: {args.top_csv}")
    sel = pd.read_csv(args.top_csv, encoding="utf-8-sig", low_memory=False)
    sel = to_pair_id(sel)

    # 決定 L 與 Z 清單
    if args.formation_lengths.lower() == "all" or args.z_windows.lower() == "all":
        Ls_avail, Zs_avail = list_LZ_from_cache(Path(args.cache_root))
    else:
        Ls_avail, Zs_avail = [], []

    if args.formation_lengths.lower() == "all":
        L_list = Ls_avail
    else:
        L_list = parse_floats_list(args.formation_lengths)

    if args.z_windows.lower() == "all":
        Z_list = Zs_avail
    else:
        Z_list = parse_ints_list(args.z_windows)

    if args.trading_periods.lower() == "all":
        TP_list = sorted(sel["trading_period"].astype(str).unique().tolist())
    else:
        TP_list = [x.strip() for x in args.trading_periods.split(",") if x.strip()]

    if not L_list or not Z_list:
        print(f"[ERROR] No L or Z found. L={L_list}, Z={Z_list}")
        return

    print(f"[INFO] Sets to run: L={L_list} × Z={Z_list} (total {len(L_list)*len(Z_list)})")
    print(f"[INFO] Years: {TP_list}")
    print(f"[INFO] Grid: z_entry={args.grid_z_entry} z_exit={args.grid_z_exit} time_stop={args.grid_time_stop}")

    z_entry_grid = parse_floats_list(args.grid_z_entry)
    z_exit_grid = parse_floats_list(args.grid_z_exit)
    tstop_grid_weeks = parse_time_stop_grid(args.grid_time_stop)

    # 平行化跑每個 Set
    tasks: List[Tuple[float, int]] = [(L, Z) for L in L_list for Z in Z_list]

    def run_one(L: float, Z: int) -> Optional[BestResult]:
        out_dir = out_root / f"L{int(round(L*100)):03d}_Z{int(Z):03d}"
        ensure_dir(out_dir)
        print(f"[INFO] Running Set L={L} Z={Z} ...")
        best = run_set(
            L=L, Z=Z,
            sel_df=sel,
            loader_root=args.cache_root,
            price_type=args.price_type,
            trading_periods=TP_list,
            z_entry_grid=z_entry_grid,
            z_exit_grid=z_exit_grid,
            tstop_grid_weeks=tstop_grid_weeks,
            cost_bps=float(args.cost_bps),
            capital=float(args.capital),
            n_pairs_cap=int(args.n_pairs_cap),
            ignore_selection_formation=bool(args.ignore_selection_formation),
            log_prefix=f"[L={L} Z={Z}] "
        )
        if best is None:
            print(f"[WARN] Set L={L} Z={Z} produced no result.")
            return None

        # 保存最佳 Set 的成果
        best_params = dict(
            formation_length=float(best.L),
            z_window=int(best.Z),
            z_entry=float(best.z_entry),
            z_exit=float(best.z_exit),
            time_stop=("none" if best.time_stop_weeks is None else f"{best.time_stop_weeks}w"),
            cost_bps=float(best.cost_bps),
            cum_return=float(best.cum_return),
            ann_return=float(best.ann_return),
            ann_vol=float(best.ann_vol),
            sharpe=float(best.sharpe),
            max_drawdown=float(best.max_drawdown),
            total_trades=int(best.total_trades),
            win_rate=float(best.win_rate) if best.win_rate == best.win_rate else None,
            avg_duration_days=float(best.avg_duration_days) if best.avg_duration_days == best.avg_duration_days else None,
            profit_factor=(float(best.profit_factor) if np.isfinite(best.profit_factor) else "inf"),
            no_trade_years=int(best.no_trade_years),
            years_covered=list(best.years_covered),
        )
        with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)

        # 全期 equity curve
        eq_df = pd.DataFrame({"date": best.equity_full.index, "equity": best.equity_full.values})
        eq_df.to_csv(out_dir / "equity_curve_full.csv", index=False, encoding="utf-8-sig")

        # 年度結果
        yr_df = pd.DataFrame(best.yearly_rows)
        yr_df.to_csv(out_dir / "yearly_metrics.csv", index=False, encoding="utf-8-sig")

        # 無交易年度統計
        with open(out_dir / "no_trade_years.txt", "w", encoding="utf-8") as f:
            f.write(f"no_trade_years={best.no_trade_years} / total_years={len(best.years_covered)}\n")

        return best

    best_list: List[BestResult] = Parallel(n_jobs=int(args.n_jobs), backend=args.backend)(
        delayed(run_one)(L, Z) for (L, Z) in tasks
    )
    best_list = [b for b in best_list if b is not None]

    if not best_list:
        print("[ERROR] No best results produced.")
        return

    # 螢幕列印表格
    print("\nTotal best sets and performance:")
    print_table(best_list)

    # 保存總表
    rows = []
    for r in best_list:
        rows.append(dict(
            formation_length=float(r.L),
            z_window=int(r.Z),
            z_entry=float(r.z_entry),
            z_exit=float(r.z_exit),
            time_stop=("none" if r.time_stop_weeks is None else f"{r.time_stop_weeks}w"),
            cost_bps=float(r.cost_bps),
            cum_return=float(r.cum_return),
            ann_return=float(r.ann_return),
            ann_vol=float(r.ann_vol),
            sharpe=float(r.sharpe),
            max_drawdown=float(r.max_drawdown),
            total_trades=int(r.total_trades),
            win_rate=float(r.win_rate) if r.win_rate == r.win_rate else np.nan,
            avg_duration_days=float(r.avg_duration_days) if r.avg_duration_days == r.avg_duration_days else np.nan,
            profit_factor=(float(r.profit_factor) if np.isfinite(r.profit_factor) else np.inf),
            no_trade_years=int(r.no_trade_years),
            years=",".join(r.years_covered)
        ))
    best_df = pd.DataFrame(rows).sort_values(["formation_length","z_window"]).reset_index(drop=True)
    best_csv = summary_dir / "best_sets.csv"
    best_df.to_csv(best_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved summary CSV: {best_csv.resolve()}")


if __name__ == "__main__":
    main()