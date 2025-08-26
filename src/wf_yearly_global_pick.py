#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wf_yearly_global_pick.py (parallelized)
- 年度走動式（walk-forward）全域唯一最佳：
  每個訓練窗（前 n 年）在「所有 L×Z × 參數格點」中並行搜尋，選出唯一最佳 (L*,Z*,θ*)，
  只用它回測下一年的 OOS。逐年滾動，輸出每年一行與全期總結。
- 平行化：訓練窗內以 (L,Z) 為單位平行搜尋（joblib loky/threading）。
- 不修改核心語義：T+1 收盤成交（PnL 用 pos.shift(1) × r），beta 用 beta.shift(1)。
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
    out: List[Optional[int]] = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if t in ("none", "nan", ""):
            out.append(None)
        else:
            out.append(int(float(t)))
    return out

def to_pair_id(df: pd.DataFrame) -> pd.DataFrame:
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
    r = returns.dropna()
    if len(r) == 0:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0)
    mu = r.mean() * freq
    sd = r.std(ddof=1)
    vol = sd * sqrt(freq) if sd > 0 else 0.0
    sharpe = mu / vol if vol > 0 else 0.0
    return dict(ann_return=float(mu), ann_vol=float(vol), sharpe=float(sharpe))

def max_drawdown_curve(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


# ===== 部位與面板 =====

def build_positions(z: pd.DataFrame, z_entry: float, z_exit: float, time_stop_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    z_signal: pd.DataFrame
    r_pair: pd.DataFrame
    beta_abs: pd.DataFrame
    pair_ids: List[str]
    w_per_pair: float

def prepare_year_panel(loader: RollingCacheLoader,
                       pair_ids: List[str],
                       tp_start: str,
                       tp_end: str,
                       price_type: str) -> Optional[YearPanel]:
    date_range = (tp_start, tp_end)
    panel_z  = loader.load_panel(pair_ids, fields=("z",),  date_range=date_range, join="outer", allow_missing=True)
    panel_b  = loader.load_panel(pair_ids, fields=("beta",),date_range=date_range, join="outer", allow_missing=True)
    panel_px = loader.load_panel(pair_ids, fields=("px",),  date_range=date_range, join="outer", allow_missing=True)
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
              capital: float) -> Tuple[pd.Series, Dict[str, float], Dict[str, float], List[float], List[int]]:
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

    # 逐筆交易（pair 級）
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
    return ret, m, trade_stats, trade_pnls, trade_durs


# ===== 年度走動式（全域唯一最佳；訓練窗並行搜尋） =====

@dataclass
class YearRow:
    year: str
    L: float
    Z: int
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

def print_table(rows: List[YearRow], total: Dict[str, float]):
    headers = ["year","L","Z","z_entry","z_exit","time_stop","ann_return","ann_vol","sharpe","max_drawdown",
               "total_trades","win_rate","avg_duration_days","profit_factor"]
    def pct(x): return f"{x:.2%}" if x == x and np.isfinite(x) else "NA"
    def num(x,n=1): return f"{x:.{n}f}" if x == x and np.isfinite(x) else "NA"
    data = []
    for r in rows:
        data.append([
            str(int(float(r.year))), num(r.L,1), f"{int(r.Z)}",
            num(r.z_entry,1), num(r.z_exit,1),
            ("none" if r.time_stop_weeks is None else f"{r.time_stop_weeks}w"),
            pct(r.ann_return), pct(r.ann_vol), num(r.sharpe,2), pct(r.max_drawdown),
            f"{int(r.total_trades)}", pct(r.win_rate), num(r.avg_duration_days,1),
            ("inf" if not np.isfinite(r.profit_factor) else f"{r.profit_factor:.2f}")
        ])
    widths=[]
    for j,h in enumerate(headers):
        w = len(h)
        for i in range(len(data)):
            w = max(w, len(data[i][j]))
        widths.append(w+1)
    print("\nWF yearly global best (one L×Z×θ per year):")
    print("".join(h.ljust(widths[j]) for j,h in enumerate(headers)).rstrip())
    print("-"*sum(widths))
    for row in data:
        print("".join(row[j].rjust(widths[j]) for j in range(len(headers))).rstrip())
    def num2(x,n=2): return f"{x:.{n}f}" if x==x and np.isfinite(x) else "NA"
    print(f"Total OOS -> Sharpe={num2(total['sharpe'],2)} AnnRet={pct(total['ann_return'])} AnnVol={pct(total['ann_vol'])} MDD={pct(total['max_drawdown'])}")


def main():
    ap = argparse.ArgumentParser(description="WF yearly re-optimization with single global best (L×Z×θ) per year [parallelized].")
    ap.add_argument("--top-csv", type=str, default="cache/top_pairs_annual.csv")
    ap.add_argument("--cache-root", type=str, default="cache/rolling_cache_weekly_v1")
    ap.add_argument("--price-type", type=str, default="log", choices=["log","raw"])
    ap.add_argument("--formation-lengths", type=str, default="all")
    ap.add_argument("--z-windows", type=str, default="all")
    ap.add_argument("--trading-periods", type=str, default="all")
    ap.add_argument("--train-periods", type=int, default=1)
    ap.add_argument("--grid-z-entry", type=str, default="0.5,1.0,1.5,2.0,2.5")
    ap.add_argument("--grid-z-exit", type=str, default="0.0,0.5,1.0")
    ap.add_argument("--grid-time-stop", type=str, default="none,6")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--n-pairs-cap", type=int, default=60)
    ap.add_argument("--ignore-selection-formation", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=8, help="Parallel jobs for L×Z search per training window.")
    ap.add_argument("--backend", type=str, default="loky", choices=["loky","threading"], help="Joblib backend.")
    ap.add_argument("--out-dir", type=str, default="reports/wf_yearly_global_pick")
    args = ap.parse_args()

    out_root = Path(args.out_dir); ensure_dir(out_root)

    # Selection 與年度序列
    sel = pd.read_csv(args.top_csv, encoding="utf-8-sig", low_memory=False)
    sel = to_pair_id(sel)
    years_all = (sorted(sel["trading_period"].astype(str).unique().tolist())
                 if args.trading_periods.lower()=="all"
                 else [x.strip() for x in args.trading_periods.split(",") if x.strip()])
    if len(years_all) <= args.train_periods:
        print(f"[ERROR] Not enough periods: total={len(years_all)} <= train_periods={args.train_periods}")
        return

    # L、Z 空間
    if args.formation_lengths.lower()=="all" or args.z_windows.lower()=="all":
        Ls_avail, Zs_avail = list_LZ_from_cache(Path(args.cache_root))
    else:
        Ls_avail, Zs_avail = [], []
    L_list = Ls_avail if args.formation_lengths.lower()=="all" else [float(x.strip()) for x in args.formation_lengths.split(",") if x.strip()]
    Z_list = Zs_avail if args.z_windows.lower()=="all" else [int(x.strip()) for x in args.z_windows.split(",") if x.strip()]
    if not L_list or not Z_list:
        print(f"[ERROR] No L or Z found. L={L_list}, Z={Z_list}")
        return

    # 參數格
    z_entry_grid = parse_floats_list(args.grid_z_entry)
    z_exit_grid  = parse_floats_list(args.grid_z_exit)
    tstop_grid   = parse_time_stop_grid(args.grid_time_stop)

    print(f"[INFO] Years={years_all} | train_periods={args.train_periods}")
    print(f"[INFO] Parallel search per window with n_jobs={args.n_jobs}, backend={args.backend}")
    print(f"[INFO] Space: L={L_list} × Z={Z_list} × z_entry={z_entry_grid} × z_exit={z_exit_grid} × tstop={tstop_grid}")

    # 工具：給年度產生 pair_id 清單
    def get_pairs_for_year(year: str, L: float) -> List[str]:
        if args.ignore_selection_formation:
            g = sel[sel["trading_period"].astype(str) == year].copy()
        else:
            g = sel[(sel["trading_period"].astype(str) == year) &
                    (pd.to_numeric(sel["formation_length"], errors="coerce") == float(L))].copy()
        if g.empty:
            return []
        if "rank_final" in g.columns:
            g = g.sort_values(["rank_final"], ascending=True)
        return g["pair_id"].dropna().astype(str).unique().tolist()[:int(args.n_pairs_cap)]

    # Worker：在訓練窗內計算某個 L×Z 的最佳 θ 與分數
    def search_best_for_LZ(L: float, Z: int, train_years: List[str]) -> Optional[Tuple[Tuple[float,int,float,float,Optional[int]], Tuple[float,float,float]]]:
        # 先建立 Loader（此 worker 專用）
        loader = RollingCacheLoader(root=args.cache_root, price_type=args.price_type,
                                    formation_length=float(L), z_window=int(Z), log_level="ERROR")
        panels = []
        for y in train_years:
            pids = get_pairs_for_year(y, L)
            if not pids:
                continue
            p = prepare_year_panel(loader, pids, f"{y}-01-01", f"{y}-12-31", price_type=args.price_type)
            if p is not None:
                panels.append((y, p))
        if not panels:
            return None

        best_score = None
        best_tuple = None
        for ze in z_entry_grid:
            for zx in z_exit_grid:
                for tsw in tstop_grid:
                    t_days = None if tsw is None else int(ceil(float(tsw)*5.0))
                    rets = []
                    for _, p in panels:
                        ret_y, _, _, _, _ = eval_year(p, ze, zx, t_days, float(args.cost_bps), float(args.capital))
                        rets.append(ret_y)
                    if not rets:
                        continue
                    r_full = pd.concat(rets).sort_index()
                    eq = (1.0 + r_full.fillna(0.0)).cumprod()
                    m  = ann_metrics(r_full)
                    cum_ret = float(eq.iloc[-1]-1.0) if len(eq) else 0.0
                    score = (round(m["sharpe"],10), round(cum_ret,10), -round(m["ann_vol"],10))
                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best_tuple = (float(L), int(Z), float(ze), float(zx), (int(tsw) if tsw is not None else None))
        if best_tuple is None:
            return None
        return best_tuple, best_score

    rows: List[YearRow] = []
    all_oos_returns: List[pd.Series] = []

    # 逐年走動
    for i in range(args.train_periods, len(years_all)):
        train_years = years_all[i-args.train_periods:i]
        test_year   = years_all[i]

        # 訓練窗：平行搜尋所有 L×Z
        tasks = [(L, Z) for L in L_list for Z in Z_list]
        results = Parallel(n_jobs=int(args.n_jobs), backend=args.backend)(
            delayed(search_best_for_LZ)(L, Z, train_years) for (L, Z) in tasks
        )
        # 取全域唯一最佳
        best_tuple = None
        best_score = None
        for res in results:
            if res is None:
                continue
            tpl, score = res
            if (best_score is None) or (score > best_score):
                best_tuple, best_score = tpl, score

        if best_tuple is None:
            print(f"[WARN] No valid (L,Z) in training {train_years}. Skip test year {test_year}.")
            continue

        Lstar, Zstar, ze_star, zx_star, tsw_star = best_tuple
        t_days_star = None if tsw_star is None else int(ceil(float(tsw_star)*5.0))
        # 測試窗 OOS
        pids_test = get_pairs_for_year(test_year, Lstar)
        if not pids_test:
            print(f"[WARN] No pairs for test year {test_year} at L={Lstar}. Skip.")
            continue
        loader = RollingCacheLoader(root=args.cache_root, price_type=args.price_type,
                                    formation_length=float(Lstar), z_window=int(Zstar), log_level="ERROR")
        p_test = prepare_year_panel(loader, pids_test, f"{test_year}-01-01", f"{test_year}-12-31", price_type=args.price_type)
        if p_test is None:
            print(f"[WARN] No panel for test year {test_year} at L={Lstar},Z={Zstar}. Skip.")
            continue

        ret_oos, m_y, tstats_y, _, _ = eval_year(p_test, ze_star, zx_star, t_days_star, float(args.cost_bps), float(args.capital))
        all_oos_returns.append(ret_oos)

        rows.append(YearRow(
            year=str(test_year), L=float(Lstar), Z=int(Zstar),
            z_entry=float(ze_star), z_exit=float(zx_star),
            time_stop_weeks=(int(tsw_star) if tsw_star is not None else None),
            ann_return=float(m_y["ann_return"]), ann_vol=float(m_y["ann_vol"]), sharpe=float(m_y["sharpe"]),
            max_drawdown=float(m_y["max_drawdown"]),
            total_trades=int(tstats_y["total_trades"]),
            win_rate=float(tstats_y["win_rate"]) if tstats_y["win_rate"] == tstats_y["win_rate"] else np.nan,
            avg_duration_days=float(tstats_y["avg_duration_days"]) if tstats_y["avg_duration_days"] == tstats_y["avg_duration_days"] else np.nan,
            profit_factor=float(tstats_y["profit_factor"]) if tstats_y["profit_factor"] == tstats_y["profit_factor"] else np.nan
        ))

    if not rows:
        print("[ERROR] No OOS rows produced.")
        return

    # 全期 OOS 指標
    ret_full = pd.concat(all_oos_returns).sort_index()
    eq_full  = (1.0 + ret_full.fillna(0.0)).cumprod()
    dd_full  = max_drawdown_curve(eq_full)
    m_full   = ann_metrics(ret_full)
    m_full["max_drawdown"] = float(dd_full.min()) if len(dd_full) else 0.0

    # 螢幕列印與輸出
    print_table(rows, m_full)
    df_rows = pd.DataFrame([r.__dict__ for r in rows]).sort_values("year")
    df_rows.to_csv(out_root / "global_reopt_oos_yearly.csv", index=False, encoding="utf-8-sig")
    eq_df = pd.DataFrame({"date": ret_full.index, "ret": ret_full.values})
    eq_df["equity"] = eq_full.reindex(ret_full.index).values
    eq_df.to_csv(out_root / "global_reopt_oos_returns.csv", index=False, encoding="utf-8-sig")
    with open(out_root / "global_reopt_summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(
            ann_return=float(m_full["ann_return"]), ann_vol=float(m_full["ann_vol"]),
            sharpe=float(m_full["sharpe"]), max_drawdown=float(m_full["max_drawdown"])
        ), f, ensure_ascii=False, indent=2)

    print(f"[INFO] Outputs saved under: {out_root.resolve()}")


if __name__ == "__main__":
    main()