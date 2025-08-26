#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wf_yearly_reopt.py
- 典型年度走動式（walk-forward yearly re-optimization）
- 訓練窗：以前 n 年（--train-periods）；在訓練窗內「直接用訓練資料」選唯一最佳 θ*（不做 CV）
- 測試窗：用 θ* 交易下一年，取 OOS 指標
- 逐年重複，列印 2015–2019（或全樣本）每年最佳參數與績效，並輸出全期累計績效與檔案
- 註解：繁體中文；print/log 英文
"""

import argparse
import json
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# 匯入週頻快取 Loader
try:
    from .cache_loader import RollingCacheLoader
except Exception:
    from src.cache_loader import RollingCacheLoader


# ===== 工具 =====

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_floats_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_ints_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_time_stop_grid(s: str) -> List[Optional[int]]:
    """grid 解析：支援 'none' 回傳 None 表示無時間停損（其餘為週數整數）。"""
    out: List[Optional[int]] = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if t in ("none", "nan", ""):
            out.append(None)
        else:
            out.append(int(float(t)))
    return out

def to_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    """保證 pair_id 存在（方向化，stock1__stock2）。"""
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
    """掃描 cache 根目錄取得可用的 L 與 Z。"""
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


# ===== 部位邏輯與面板 =====

def build_positions(z: pd.DataFrame, z_entry: float, z_exit: float, time_stop_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """由 z 生成部位與持有日數（外部已 shift，避免前視）。"""
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


@dataclass
class YearPanel:
    dates: pd.DatetimeIndex
    z_signal: pd.DataFrame   # [dates × pairs]（已 shift(1)）
    r_pair: pd.DataFrame     # [dates × pairs]
    beta_abs: pd.DataFrame   # [dates × pairs] 用於成本
    pair_ids: List[str]
    w_per_pair: float

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

    dates = panel_z.index.union(panel_b.index).union(panel_px.index).sort_values()
    z = panel_z.reindex(dates)["z"]
    beta = panel_b.reindex(dates)["beta"]

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
    z_signal = z.shift(1)  # 不前視

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
              time_stop_days: Optional[int],
              cost_bps: float,
              capital: float) -> Tuple[pd.Series, Dict[str, float], Dict[str, float], Dict[str, int], List[float], List[int]]:
    """
    評估單年度，回傳：
    - ret（日報酬，含成本）
    - metrics：ann_return/ann_vol/sharpe/max_drawdown
    - trade_stats（逐筆）：total_trades, win_rate, avg_duration_days, profit_factor
    - portfolio_stats：event_days（當天任一對 |Δpos|>0）
    - trade_pnls, trade_durs（逐筆 PnL 與持有天）
    """
    zsig = year_panel.z_signal
    r_pair = year_panel.r_pair
    beta_abs = year_panel.beta_abs
    w = year_panel.w_per_pair
    cap = float(capital)

    tstop = time_stop_days if time_stop_days is not None else 10**9

    pos, days = build_positions(zsig, z_entry=z_entry, z_exit=z_exit, time_stop_days=tstop)

    pnl_ex = (pos.shift(1) * r_pair * (w * cap)).sum(axis=1)

    dpos = pos.fillna(0.0).diff().abs()
    traded_notional = ((w * cap) * (1.0 + beta_abs) * dpos).sum(axis=1)
    cost = traded_notional * (float(cost_bps) / 10000.0)

    pnl_net = pnl_ex - cost
    ret = pnl_net / cap
    equity = (1.0 + ret.fillna(0.0)).cumprod()
    dd = max_drawdown_curve(equity)

    m = ann_metrics(ret)
    m["max_drawdown"] = float(dd.min()) if len(dd) else 0.0

    # 逐筆交易統計（pair 級）
    trade_pnls: List[float] = []
    trade_durs: List[int] = []
    cost_rate = float(cost_bps) / 10000.0

    for pid in pos.columns:
        pos_s = pos[pid]
        r_s = r_pair[pid]
        beta_abs_s = beta_abs[pid]
        pnl_ex_s = (pos_s.shift(1) * r_s * (w * cap))
        cost_s = ((w * cap) * (1.0 + beta_abs_s) * pos_s.fillna(0.0).diff().abs()) * cost_rate
        pnl_net_s = pnl_ex_s - cost_s
        holding = pos_s.fillna(0.0) != 0.0
        if holding.any():
            h = holding.astype(int).values
            idx = np.where(np.diff(np.r_[0, h, 0]) != 0)[0]
            for j in range(0, len(idx), 2):
                start = idx[j]
                end = idx[j + 1] - 1
                if end >= start + 1:
                    pnl_trade = float(pnl_net_s.iloc[start + 1: end + 1].sum())
                    dur_days = int(end - start + 1)
                    trade_pnls.append(pnl_trade)
                    trade_durs.append(dur_days)

    total_trades = len(trade_pnls)
    sum_win = sum(p for p in trade_pnls if p > 0)
    sum_loss = sum(-p for p in trade_pnls if p < 0)
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = (wins / total_trades) if total_trades > 0 else np.nan
    profit_factor = (sum_win / sum_loss) if sum_loss > 0 else (np.inf if sum_win > 0 else np.nan)
    avg_duration_days = (np.mean(trade_durs) if trade_durs else np.nan)

    trade_stats = dict(
        total_trades=int(total_trades),
        win_rate=float(win_rate) if win_rate == win_rate else np.nan,
        avg_duration_days=float(avg_duration_days) if avg_duration_days == avg_duration_days else np.nan,
        profit_factor=float(profit_factor) if profit_factor not in (np.inf, -np.inf) else np.inf
    )

    portfolio_stats = dict(
        event_days=int((dpos.sum(axis=1) > 0).sum())
    )

    return ret, m, trade_stats, portfolio_stats, trade_pnls, trade_durs


# ===== 訓練窗選唯一最佳參數（直接用訓練資料） =====

def select_theta_train_is(train_years: List[str],
                          year_panels: Dict[str, YearPanel],
                          z_entry_grid: List[float],
                          z_exit_grid: List[float],
                          tstop_grid_weeks: List[Optional[int]],
                          cost_bps: float,
                          capital: float) -> Tuple[float, float, Optional[int]]:
    """
    在訓練窗內「直接用訓練資料」選唯一最佳 θ*（Sharpe → 累積報酬 → 年化波動）。
    """
    cand = []
    for ze in z_entry_grid:
        for zx in z_exit_grid:
            for tsw in tstop_grid_weeks:
                time_stop_days = None if tsw is None else int(ceil(float(tsw) * 5.0))
                rets = []
                for y in train_years:
                    panel = year_panels.get(y)
                    if panel is None:
                        continue
                    ret_y, _, _, _, _, _ = eval_year(
                        year_panel=panel,
                        z_entry=float(ze),
                        z_exit=float(zx),
                        time_stop_days=time_stop_days,
                        cost_bps=float(cost_bps),
                        capital=float(capital)
                    )
                    rets.append(ret_y)
                if not rets:
                    continue
                r_full = pd.concat(rets).sort_index()
                eq_full = (1.0 + r_full.fillna(0.0)).cumprod()
                m = ann_metrics(r_full)
                m["max_drawdown"] = float(max_drawdown_curve(eq_full).min()) if len(eq_full) else 0.0
                cum_return = float(eq_full.iloc[-1] - 1.0) if len(eq_full) else 0.0
                cand.append(dict(ze=ze, zx=zx, tsw=tsw,
                                 sharpe=float(m["sharpe"]),
                                 cum_return=cum_return,
                                 ann_vol=float(m["ann_vol"])))
    if not cand:
        return 2.0, 0.5, None
    # 排序：Sharpe 降冪 → cum_return 降冪 → ann_vol 升冪
    best = sorted(cand, key=lambda r: (round(r["sharpe"], 10), round(r["cum_return"], 10), -round(r["ann_vol"], 10)),
                  reverse=True)[0]
    return float(best["ze"]), float(best["zx"]), (int(best["tsw"]) if best["tsw"] is not None else None)


# ===== 螢幕輸出（對齊且年份無小數、time_stop 右對齊） =====

@dataclass
class YearRow:
    year: str
    z_entry: float
    z_exit: float
    time_stop_weeks: Optional[int]
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    avg_duration_days: float
    profit_factor: float

def print_oos_table(set_name: str, rows: List[YearRow], total_metrics: Dict[str, float]):
    headers = ["year","z_entry","z_exit","time_stop","ann_return","ann_vol","sharpe","max_drawdown",
               "total_trades","win_rate","avg_duration_days","profit_factor"]

    def fmt_year(y) -> str:
        try:
            yf = float(y)
            yi = int(round(yf))
            return str(yi) if abs(yf - yi) < 1e-9 else str(y)
        except Exception:
            s = str(y)
            return s.split(".")[0] if s.endswith(".0") else s

    def pct(x): return f"{x:.2%}" if x == x and np.isfinite(x) else "NA"
    def num(x, n=1): return f"{x:.{n}f}" if x == x and np.isfinite(x) else "NA"

    data: List[List[str]] = []
    for r in rows:
        data.append([
            fmt_year(r.year),
            num(r.z_entry,1),
            num(r.z_exit,1),
            ("none" if r.time_stop_weeks is None else f"{r.time_stop_weeks}w"),
            pct(r.ann_return),
            pct(r.ann_vol),
            num(r.sharpe,2),
            pct(r.max_drawdown),
            f"{r.total_trades}",
            pct(r.win_rate),
            num(r.avg_duration_days,1),
            ("inf" if not np.isfinite(r.profit_factor) else f"{r.profit_factor:.2f}")
        ])

    widths = []
    for j, h in enumerate(headers):
        w = len(h)
        for i in range(len(data)):
            w = max(w, len(data[i][j]))
        widths.append(w + 1)

    print(f"\nWF yearly re-optimization ({set_name}):")
    header = "".join(h.ljust(widths[j]) for j, h in enumerate(headers)).rstrip()
    print(header)
    print("-" * len(header))
    for row in data:
        line = "".join(row[j].rjust(widths[j]) for j in range(len(headers))).rstrip()
        print(line)

    def num2(x, n=2): return f"{x:.{n}f}" if x == x and np.isfinite(x) else "NA"
    total_line = f"Total OOS -> Sharpe={num2(total_metrics['sharpe'],2)} AnnRet={pct(total_metrics['ann_return'])} AnnVol={pct(total_metrics['ann_vol'])} MDD={pct(total_metrics['max_drawdown'])}"
    print(total_line)


# ===== 主流程（每個 Set：L×Z） =====

def run_set(L: float,
            Z: int,
            sel_df: pd.DataFrame,
            loader_root: str,
            price_type: str,
            years_all: List[str],
            train_periods: int,
            z_entry_grid: List[float],
            z_exit_grid: List[float],
            tstop_grid_weeks: List[Optional[int]],
            cost_bps: float,
            capital: float,
            n_pairs_cap: int,
            ignore_selection_formation: bool,
            out_root: Path) -> Optional[Path]:
    """單一 Set 的年度走動式流程；回傳輸出目錄路徑。"""
    set_dir = out_root / f"L{int(round(L*100)):03d}_Z{int(Z):03d}"
    ensure_dir(set_dir)

    loader = RollingCacheLoader(
        root=loader_root, price_type=price_type,
        formation_length=float(L), z_window=int(Z),
        log_level="ERROR"
    )

    # 準備每年的面板
    year_panels: Dict[str, YearPanel] = {}
    for y in years_all:
        if ignore_selection_formation:
            g = sel_df[sel_df["trading_period"].astype(str) == y].copy()
        else:
            g = sel_df[(sel_df["trading_period"].astype(str) == y) &
                       (pd.to_numeric(sel_df["formation_length"], errors="coerce") == float(L))].copy()
        if g.empty:
            year_panels[y] = None
            continue
        if "rank_final" in g.columns:
            g = g.sort_values(["rank_final"], ascending=True)
        pair_ids = g["pair_id"].dropna().astype(str).unique().tolist()[:int(n_pairs_cap)]
        if not pair_ids:
            year_panels[y] = None
            continue
        t_start = str(g["trading_start"].iloc[0]) if "trading_start" in g.columns else f"{y}-01-01"
        t_end   = str(g["trading_end"].iloc[0])   if "trading_end" in g.columns   else f"{y}-12-31"
        panel = prepare_year_panel(loader, pair_ids, t_start, t_end, price_type=price_type)
        year_panels[y] = panel

    # 走動式
    rows: List[YearRow] = []
    all_oos_rets: List[pd.Series] = []

    for i in range(train_periods, len(years_all)):
        train_years = [y for y in years_all[i-train_periods:i] if year_panels.get(y) is not None]
        test_year = years_all[i]
        panel_test = year_panels.get(test_year)
        if not train_years or panel_test is None:
            continue

        # 訓練窗選唯一最佳 θ*
        ze, zx, tsw = select_theta_train_is(
            train_years=train_years,
            year_panels=year_panels,
            z_entry_grid=z_entry_grid,
            z_exit_grid=z_exit_grid,
            tstop_grid_weeks=tstop_grid_weeks,
            cost_bps=float(cost_bps),
            capital=float(capital)
        )
        tstop_days = None if tsw is None else int(ceil(float(tsw) * 5.0))

        # 測試窗（OOS）
        ret_oos, m_y, tstats_y, _, _, _ = eval_year(
            year_panel=panel_test,
            z_entry=ze, z_exit=zx, time_stop_days=tstop_days,
            cost_bps=float(cost_bps), capital=float(capital)
        )
        all_oos_rets.append(ret_oos)

        rows.append(YearRow(
            year=str(test_year),
            z_entry=float(ze),
            z_exit=float(zx),
            time_stop_weeks=(int(tsw) if tsw is not None else None),
            ann_return=float(m_y["ann_return"]),
            ann_vol=float(m_y["ann_vol"]),
            sharpe=float(m_y["sharpe"]),
            max_drawdown=float(m_y["max_drawdown"]),
            total_trades=int(tstats_y["total_trades"]),
            win_rate=float(tstats_y["win_rate"]) if tstats_y["win_rate"] == tstats_y["win_rate"] else np.nan,
            avg_duration_days=float(tstats_y["avg_duration_days"]) if tstats_y["avg_duration_days"] == tstats_y["avg_duration_days"] else np.nan,
            profit_factor=float(tstats_y["profit_factor"]) if tstats_y["profit_factor"] == tstats_y["profit_factor"] else np.nan
        ))

    if not rows:
        return None

    # 全期 OOS 聚合
    ret_full = pd.concat(all_oos_rets).sort_index()
    eq_full = (1.0 + ret_full.fillna(0.0)).cumprod()
    dd_full = max_drawdown_curve(eq_full)
    m_full = ann_metrics(ret_full)
    m_full["max_drawdown"] = float(dd_full.min()) if len(dd_full) else 0.0

    # 輸出檔案
    df_rows = pd.DataFrame([r.__dict__ for r in rows]).sort_values("year")
    df_rows.to_csv(set_dir / "reopt_oos_yearly.csv", index=False, encoding="utf-8-sig")
    eq_df = pd.DataFrame({"date": ret_full.index, "ret": ret_full.values})
    eq_df["equity"] = eq_full.reindex(ret_full.index).values
    eq_df.to_csv(set_dir / "reopt_oos_returns.csv", index=False, encoding="utf-8-sig")
    summary = dict(
        formation_length=float(L),
        z_window=int(Z),
        ann_return=float(m_full["ann_return"]),
        ann_vol=float(m_full["ann_vol"]),
        sharpe=float(m_full["sharpe"]),
        max_drawdown=float(m_full["max_drawdown"])
    )
    with open(set_dir / "reopt_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 螢幕列印
    print_oos_table(set_name=f"L={L}, Z={Z}", rows=rows, total_metrics=summary)

    return set_dir


def main():
    ap = argparse.ArgumentParser(description="Walk-forward yearly re-optimization (unique-best on training window).")
    ap.add_argument("--top-csv", type=str, default="cache/top_pairs_annual.csv", help="Selection CSV.")
    ap.add_argument("--cache-root", type=str, default="cache/rolling_cache_weekly_v1", help="Weekly cache root.")
    ap.add_argument("--price-type", type=str, default="log", choices=["log","raw"], help="Price type.")
    ap.add_argument("--formation-lengths", type=str, default="all", help='Comma floats in years or "all".')
    ap.add_argument("--z-windows", type=str, default="all", help='Comma ints in weeks or "all".')
    ap.add_argument("--trading-periods", type=str, default="all", help='Comma years or "all".')
    ap.add_argument("--train-periods", type=int, default=1, help="Number of preceding years for training.")
    ap.add_argument("--grid-z-entry", type=str, default="0.5,1.0,1.5,2.0,2.5", help="Grid for z_entry.")
    ap.add_argument("--grid-z-exit", type=str, default="0.0,0.5,1.0", help="Grid for z_exit.")
    ap.add_argument("--grid-time-stop", type=str, default="none,6", help='Grid for time stop (weeks); supports "none".')
    ap.add_argument("--cost-bps", type=float, default=5.0, help="Per-leg, one-way cost in bps.")
    ap.add_argument("--capital", type=float, default=1_000_000.0, help="Capital for scaling.")
    ap.add_argument("--n-pairs-cap", type=int, default=60, help="Max pairs per year.")
    ap.add_argument("--ignore-selection-formation", action="store_true", help="Reuse same pair list across L.")
    ap.add_argument("--n-jobs", type=int, default=8, help="Parallel sets.")
    ap.add_argument("--backend", type=str, default="loky", choices=["loky","threading"], help="Joblib backend.")
    ap.add_argument("--out-dir", type=str, default="reports/wf_yearly_reopt", help="Output root.")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    summary_dir = out_root / "_summary"
    ensure_dir(summary_dir)

    # 讀 selection 與年度序列
    print(f"[INFO] Loading selection: {args.top_csv}")
    sel = pd.read_csv(args.top_csv, encoding="utf-8-sig", low_memory=False)
    sel = to_pair_id(sel)
    if args.trading_periods.lower() == "all":
        years_all = sorted(sel["trading_period"].astype(str).unique().tolist())
    else:
        years_all = [x.strip() for x in args.trading_periods.split(",") if x.strip()]
    if len(years_all) <= args.train_periods:
        print(f"[ERROR] Not enough periods: total={len(years_all)} <= train_periods={args.train_periods}")
        return

    # L 與 Z
    if args.formation_lengths.lower() == "all" or args.z_windows.lower() == "all":
        Ls_avail, Zs_avail = list_LZ_from_cache(Path(args.cache_root))
    else:
        Ls_avail, Zs_avail = [], []
    L_list = Ls_avail if args.formation_lengths.lower() == "all" else parse_floats_list(args.formation_lengths)
    Z_list = Zs_avail if args.z_windows.lower() == "all" else parse_ints_list(args.z_windows)
    if not L_list or not Z_list:
        print(f"[ERROR] No L or Z found. L={L_list}, Z={Z_list}")
        return

    # Grid
    z_entry_grid = parse_floats_list(args.grid_z_entry)
    z_exit_grid = parse_floats_list(args.grid_z_exit)
    tstop_grid_weeks = parse_time_stop_grid(args.grid_time_stop)

    print(f"[INFO] WF yearly reopt Sets: L={L_list} × Z={Z_list} | train_periods={args.train_periods} | years={years_all}")
    print(f"[INFO] Grid: z_entry={z_entry_grid} z_exit={z_exit_grid} time_stop={tstop_grid_weeks}")

    # 平行執行各 Set
    tasks = [(L, Z) for L in L_list for Z in Z_list]

    def runner(L, Z):
        return run_set(L, Z, sel, args.cache_root, args.price_type,
                       years_all, int(args.train_periods),
                       z_entry_grid, z_exit_grid, tstop_grid_weeks,
                       float(args.cost_bps), float(args.capital),
                       int(args.n_pairs_cap), bool(args.ignore_selection_formation),
                       out_root)

    res = Parallel(n_jobs=int(args.n_jobs), backend=args.backend)(delayed(runner)(L, Z) for (L, Z) in tasks)
    used = [str(p) for p in res if p is not None]

    # 寫 summary 檔（只記錄哪些 Set 有輸出）
    with open(summary_dir / "sets_done.json", "w", encoding="utf-8") as f:
        json.dump(dict(sets=used), f, ensure_ascii=False, indent=2)

    print(f"[INFO] Done. Outputs under: {out_root.resolve()}")


if __name__ == "__main__":
    main()