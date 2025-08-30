#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weekly_long_short_decompose.py
- 依 weekly 的 best_sets.csv（_summary 目錄）與 top_pairs_annual.csv，
  用最佳參數 per Set 重算每日部位與損益，做 Long/Short 分桶（規則同日頻）。
- 輸入：
  - --best-sets-csv：.../_summary/best_sets.csv
  - --top-csv：weekly selection（例如 cache/top_pairs_annual.csv）
  - --cache-root, --price-type
  - --n-pairs-cap：每年取前 K（與原 wrapper 一致）
- 輸出（於 --out-dir）：
  - long_short__L{L}_Z{Z}.csv
  - long_short_summary.csv
註解：繁體中文；print/log：英文
"""
import os
import sys
import math
import argparse
import logging
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import warnings; warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.core.indexes.base")

try:
    from .cache_loader import RollingCacheLoader
except Exception:
    try:
        from src.cache_loader import RollingCacheLoader
    except Exception:
        RollingCacheLoader = None

def setup_logger(level="INFO"):
    logger = logging.getLogger()
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    if "pair_id" in df.columns:
        return df
    if {"stock1","stock2"}.issubset(df.columns):
        df = df.copy()
        df["pair_id"] = df["stock1"].astype(str) + "__" + df["stock2"].astype(str)
        return df
    if "pair" in df.columns:
        df = df.copy()
        df["pair_id"] = df["pair"].astype(str)
        return df
    raise ValueError("top-csv needs 'pair_id' or ('stock1','stock2') or 'pair'.")

def lcode(L: float) -> str:
    return f"{int(round(float(L)*100)):03d}"

def ann_metrics(ret: pd.Series, k: int = 252) -> Tuple[float,float,float]:
    r = pd.Series(ret).dropna()
    if len(r) < 2: return 0.0,0.0,0.0
    mu = float(r.mean()*k); sd = float(r.std(ddof=1)*math.sqrt(k))
    sr = float((mu/sd) if sd>0 else 0.0)
    return mu, sd, sr

def mdd_from_ret(ret: pd.Series) -> float:
    eq = (1.0 + pd.Series(ret).fillna(0.0)).cumprod()
    return float((eq/eq.cummax()-1.0).min()) if len(eq)>0 else 0.0

def preload_period_data(loader: RollingCacheLoader,
                        pair_ids: List[str], t_start: str, t_end: str) -> Optional[Dict[str, Any]]:
    date_range = (t_start, t_end)
    panel_z  = loader.load_panel(pair_ids, fields=("z",),  date_range=date_range, join="outer", allow_missing=True)
    panel_b  = loader.load_panel(pair_ids, fields=("beta",), date_range=date_range, join="outer", allow_missing=True)
    panel_px = loader.load_panel(pair_ids, fields=("px",),  date_range=date_range, join="outer", allow_missing=True)
    if panel_z.empty or panel_b.empty or panel_px.empty:
        return None
    dates = panel_z.index.union(panel_b.index).union(panel_px.index).sort_values()
    z = panel_z.reindex(dates)["z"]
    beta = panel_b.reindex(dates)["beta"]
    px = panel_px.reindex(dates)
    z_sig = z.shift(1)
    b_lag = beta.shift(1)
    if loader.price_type == "log":
        rx = px["px_x"].diff()
        ry = px["px_y"].diff()
    else:
        rx = px["px_x"].pct_change()
        ry = px["px_y"].pct_change()
    r_pair = (ry - b_lag * rx)
    if z.shape[1]==0: return None
    w = 1.0 / float(z.shape[1])
    return dict(dates=dates, z_signal=z_sig, beta_abs=b_lag.abs().fillna(0.0),
                r_pair=r_pair, w=float(w))

def build_positions(z_signal: pd.DataFrame, z_entry: float, z_exit: float, tstop_days: int) -> pd.DataFrame:
    pos = z_signal.copy() * np.nan
    for c in z_signal.columns:
        s = z_signal[c]
        out = pd.Series(index=s.index, dtype="float64")
        last = 0.0; held = 0
        for i in range(len(s)):
            z = s.iloc[i]; cur = last
            if cur == 0.0:
                if pd.notna(z) and z >= z_entry: cur=-1.0; held=1
                elif pd.notna(z) and z <= -z_entry: cur=+1.0; held=1
                else: cur=0.0; held=0
            else:
                if (pd.notna(z) and abs(z) <= z_exit) or (held >= tstop_days):
                    cur=0.0; held=0
                else:
                    held+=1
            out.iloc[i]=cur; last=cur
        pos[c]=out
    return pos

def decompose_long_short(pdx: Dict[str, Any],
                         z_entry: float, z_exit: float, tstop_days: int,
                         cost_bps: float, capital: float) -> pd.DataFrame:
    dates = pdx["dates"]
    pos = build_positions(pdx["z_signal"], float(z_entry), float(z_exit), int(tstop_days))
    prev = pos.shift(1).fillna(0.0)
    r_pair = pdx["r_pair"].reindex(pos.index)
    w_notional = pdx["w"] * float(capital)
    cost_rate = float(cost_bps) / 10000.0

    pnl_ex = (prev * r_pair) * w_notional
    pnl_long_ex  = pnl_ex.where(prev > 0.0, 0.0).sum(axis=1)
    pnl_short_ex = pnl_ex.where(prev < 0.0, 0.0).sum(axis=1)

    dpos = (pos - prev).abs()
    traded_notional = ((1.0 + pdx["beta_abs"]) * dpos) * w_notional
    cost = traded_notional * cost_rate

    open_long   = ((prev == 0.0) & (pos == 1.0))
    close_long  = ((prev == 1.0) & (pos == 0.0))
    open_short  = ((prev == 0.0) & (pos == -1.0))
    close_short = ((prev == -1.0) & (pos == 0.0))
    flip_ls = ((prev == 1.0) & (pos == -1.0))
    flip_sl = ((prev == -1.0) & (pos == 1.0))

    cost_long  = cost.where(open_long | close_long, 0.0) + cost.where(flip_ls | flip_sl, 0.0) * 0.5
    cost_short = cost.where(open_short | close_short, 0.0) + cost.where(flip_ls | flip_sl, 0.0) * 0.5

    cost_long_daily  = cost_long.sum(axis=1)
    cost_short_daily = cost_short.sum(axis=1)

    pnl_long  = pnl_long_ex  - cost_long_daily
    pnl_short = pnl_short_ex - cost_short_daily
    pnl_total = pnl_long + pnl_short

    ret_long  = pnl_long  / float(capital)
    ret_short = pnl_short / float(capital)
    ret_total = pnl_total / float(capital)

    out = pd.DataFrame({"date": dates, "ret_long": ret_long.reindex(dates).fillna(0.0).values,
                        "ret_short": ret_short.reindex(dates).fillna(0.0).values,
                        "ret_total": ret_total.reindex(dates).fillna(0.0).values})
    return out

def main():
    ap = argparse.ArgumentParser(description="Weekly long/short decomposition by best parameters per Set.")
    ap.add_argument("--best-sets-csv", type=str, required=True, help="Weekly _summary/best_sets.csv")
    ap.add_argument("--top-csv", type=str, required=True, help="Weekly selection CSV (e.g., cache/top_pairs_annual.csv)")
    ap.add_argument("--cache-root", type=str, required=True, help="Weekly cache root.")
    ap.add_argument("--price-type", type=str, default="log", choices=["log","raw"], help="Price type.")
    ap.add_argument("--n-pairs-cap", type=int, default=60, help="Top-K per year.")
    ap.add_argument("--ignore-selection-formation", action="store_true", help="Reuse same selection across L.")
    ap.add_argument("--out-dir", type=str, default="", help="Output dir (default=<best-sets folder>/long_short)")
    args = ap.parse_args()

    setup_logger("INFO")
    if RollingCacheLoader is None:
        print("[ERROR] RollingCacheLoader not available."); sys.exit(2)

    if not os.path.isfile(args.best_sets_csv):
        print(f"[ERROR] best_sets.csv not found: {args.best_sets_csv}"); sys.exit(2)
    best = pd.read_csv(args.best_sets_csv, encoding="utf-8-sig")
    if "formation_length" not in best.columns or "z_window" not in best.columns:
        print("[ERROR] best_sets.csv missing required columns."); sys.exit(2)

    top = pd.read_csv(args.top_csv, encoding="utf-8-sig", low_memory=False)
    top = to_pair_id(top)

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.best_sets_csv)), "long_short")
    ensure_dir(out_dir)

    summary_rows = []
    # 逐 Set（L,Z）處理
    for _, r in best.iterrows():
        L = float(r["formation_length"]); Z = int(r["z_window"])
        ze = float(r["z_entry"]); zx=float(r["z_exit"])

        def _parse_tstop_weeks(x):
            # 支援：None / "none" / "" / "nan" / "null" → None
            #      "6w" / "6" / 6.0 → 6（週）
            if x is None:
                return None
            if isinstance(x, str):
                s = x.strip().lower()
                if s in ("none", "", "nan", "null"):
                    return None
                if s.endswith("w"):
                    s = s[:-1]
                try:
                    return int(float(s))
                except Exception:
                    return None
            # 非字串 → 嘗試轉數字
            try:
                return int(float(x))
            except Exception:
                return None

        tsw = r.get("time_stop", None)
        tstop_weeks = _parse_tstop_weeks(tsw)
        tstop_days = 999999 if (tstop_weeks is None) else int(math.ceil(float(tstop_weeks) * 5.0))

        cost_bps = float(r.get("cost_bps", 0.0))
        years = str(r.get("years","")).split(",") if "years" in r else []

        print(f"[INFO] Processing L={L} Z={Z} ...")
        loader = RollingCacheLoader(root=args.cache_root, price_type=args.price_type,
                                    formation_length=L, z_window=Z, log_level="ERROR")

        # 收集各年 pair_ids
        ret_long_all = pd.Series([], dtype="float64", index=pd.DatetimeIndex([], name="date"))
        ret_short_all = pd.Series([], dtype="float64", index=pd.DatetimeIndex([], name="date"))
        ret_total_all = pd.Series([], dtype="float64", index=pd.DatetimeIndex([], name="date"))

        if len(years) == 0:
            years = sorted(top["trading_period"].astype(str).unique().tolist())

        for y in [yy.strip() for yy in years if yy.strip()]:
            g = top[top["trading_period"].astype(str)==y].copy()
            if not args.ignore_selection_formation:
                g = g[g["formation_length"].astype(float)==L]
            if g.empty: continue
            if "rank_final" in g.columns:
                g = g.sort_values(["rank_final"], ascending=True)
            pair_ids = g["pair_id"].dropna().astype(str).unique().tolist()[:int(args.n_pairs_cap)]
            if not pair_ids: continue

            # 期間
            if "trading_start" in g.columns and "trading_end" in g.columns:
                t_start = str(pd.to_datetime(g["trading_start"].iloc[0]).date())
                t_end   = str(pd.to_datetime(g["trading_end"].iloc[0]).date())
            else:
                t_start = f"{y}-01-01"; t_end = f"{y}-12-31"

            # 缺失過濾
            missing = loader.check_missing(pair_ids)
            if missing:
                pair_ids = [p for p in pair_ids if p not in missing]
            if not pair_ids: continue

            pdx = preload_period_data(loader, pair_ids, t_start, t_end)
            if pdx is None: continue

            df = decompose_long_short(pdx, ze, zx, tstop_days, cost_bps, capital=1_000_000.0)

            df2 = df.copy()
            df2["date"] = pd.to_datetime(df2["date"])
            df2 = df2.set_index("date").sort_index()
            new_idx = df2.index # DatetimeIndex
            idx = ret_total_all.index.union(new_idx) # 兩邊同為 DatetimeIndex，不會觸發警告

        ret_long_all = ret_long_all.reindex(idx, fill_value=0.0) + df2["ret_long"].reindex(idx, fill_value=0.0)
        ret_short_all = ret_short_all.reindex(idx, fill_value=0.0) + df2["ret_short"].reindex(idx, fill_value=0.0)
        ret_total_all = ret_total_all.reindex(idx, fill_value=0.0) + df2["ret_total"].reindex(idx, fill_value=0.0)

        if len(ret_total_all) == 0:
            print(f"[WARN] No returns for L={L} Z={Z}. Skip.")
            continue

        eq_long  = (1.0 + ret_long_all.fillna(0.0)).cumprod()
        eq_short = (1.0 + ret_short_all.fillna(0.0)).cumprod()
        eq_total = (1.0 + ret_total_all.fillna(0.0)).cumprod()
        out_df = pd.DataFrame({
            "date": ret_total_all.index, "ret_long": ret_long_all.values, "ret_short": ret_short_all.values,
            "ret_total": ret_total_all.values, "equity_long": eq_long.values,
            "equity_short": eq_short.values, "equity_total": eq_total.values
        })
        fp = os.path.join(out_dir, f"long_short__L{lcode(L)}_Z{Z}.csv")
        out_df.to_csv(fp, index=False)
        print(f"[WRITE] {fp}")

        muL, sdL, srL = ann_metrics(ret_long_all, 252)
        muS, sdS, srS = ann_metrics(ret_short_all, 252)
        muT, sdT, srT = ann_metrics(ret_total_all, 252)
        mddL = mdd_from_ret(ret_long_all)
        mddS = mdd_from_ret(ret_short_all)
        mddT = mdd_from_ret(ret_total_all)
        days = len(ret_total_all)
        days_long = int((ret_long_all != 0).sum())
        days_short = int((ret_short_all != 0).sum())

        summary_rows.append(dict(
            source="weekly",
            formation_length=L, z_window=Z,
            ann_return_long=muL, ann_vol_long=sdL, sharpe_long=srL, mdd_long=mddL,
            ann_return_short=muS, ann_vol_short=sdS, sharpe_short=srS, mdd_short=mddS,
            ann_return_total=muT, ann_vol_total=sdT, sharpe_total=srT, mdd_total=mddT,
            n_days=days, n_days_long=days_long, n_days_short=days_short,
            cost_bps=cost_bps, price_type=args.price_type
        ))

    if summary_rows:
        s = pd.DataFrame(summary_rows).sort_values(["formation_length","z_window"]).reset_index(drop=True)
        s.to_csv(os.path.join(out_dir, "long_short_summary.csv"), index=False)
        print(f"[WRITE] {os.path.join(out_dir, 'long_short_summary.csv')}")
    print("Done.")

if __name__ == "__main__":
    main()