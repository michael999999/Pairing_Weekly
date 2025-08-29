#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_weekly_timeseries.py
- 讀 weekly wrapper 的 _summary/best_sets.csv（含 set_dir），對每個 set_dir 讀取全期權益（預設 equity_curve_full.csv），
  轉為回報序列並合併輸出。
- 可用 --matched-csv 只輸出 compare_daily_weekly_sets 的已配對 (L, weekly_z)。
- 註解：繁體中文；print/log：英文
"""

import os
import sys
import argparse
import logging
import glob
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

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

def parse_mapping(s: str) -> Dict[int,int]:
    """daily→weekly 映射字串，轉成 weekly→daily 對照用。"""
    w2d = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        d, w = tok.split(":")
        w2d[int(float(w.strip()))] = int(float(d.strip()))
    return w2d

def read_matched_sets(path: str) -> Set[Tuple[float,int]]:
    """讀 matched_comparison.csv，回傳 {(L, weekly_z)}。"""
    try:
        m = pd.read_csv(path)
        if not {"L","weekly_z"}.issubset(set(m.columns)):
            return set()
        out = set()
        for _, r in m.iterrows():
            out.add((float(r["L"]), int(r["weekly_z"])))
        return out
    except Exception:
        return set()

def find_equity_file(set_dir: str, preferred: Optional[str] = None) -> str:
    """在 set_dir 找權益 CSV；優先用 preferred，否則 equity_curve_full.csv / equity.csv / oos_equity.csv / equity*.csv。"""
    if preferred:
        p = os.path.join(set_dir, preferred)
        if os.path.isfile(p):
            return p
    candidates = ["equity_curve_full.csv", "equity.csv", "oos_equity.csv", "equity_curve.csv"]
    for c in candidates:
        p = os.path.join(set_dir, c)
        if os.path.isfile(p):
            return p
    g = glob.glob(os.path.join(set_dir, "equity*.csv"))
    g = [x for x in g if os.path.isfile(x)]
    return sorted(g)[0] if g else ""

def main():
    ap = argparse.ArgumentParser(description="Build weekly timeseries CSV from weekly best_sets summary.")
    ap.add_argument("--best-sets-csv", type=str, required=True,
                    help="Path to weekly _summary/best_sets.csv")
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="Filter weekly rows by cost_bps (optional).")
    ap.add_argument("--matched-csv", type=str, default=None,
                    help="Optional matched_comparison.csv to restrict (L,weekly_z).")
    ap.add_argument("--equity-filename", type=str, default=None,
                    help="If set, use this filename inside each set_dir (e.g., equity_curve_full.csv).")
    ap.add_argument("--mapping", type=str, default="21:4,42:8,63:13,126:26,252:52",
                    help="Daily→Weekly mapping (used to build family tag).")
    ap.add_argument("--out-csv", type=str, default=None,
                    help="Output CSV path (default=<best-sets folder>/ts_weekly.csv)")
    args = ap.parse_args()

    setup_logger("INFO")

    if not os.path.isfile(args.best_sets_csv):
        print(f"[ERROR] best_sets.csv not found: {args.best_sets_csv}")
        sys.exit(2)

    dfw = pd.read_csv(args.best_sets_csv, encoding="utf-8-sig", low_memory=False)
    if args.cost_bps is not None and "cost_bps" in dfw.columns:
        dfw = dfw[dfw["cost_bps"].astype(float) == float(args.cost_bps)]
    if dfw.empty:
        print("[ERROR] no weekly rows after filtering"); sys.exit(2)

    matched: Set[Tuple[float,int]] = set()
    if args.matched_csv and os.path.isfile(args.matched_csv):
        matched = read_matched_sets(args.matched_csv)
        print(f"[INFO] Matched filter loaded: {len(matched)} rows")

    w2d = parse_mapping(args.mapping)
    out_rows: List[dict] = []

    for _, r in dfw.iterrows():
        L = float(r["formation_length"])
        Ww = int(r["z_window"])
        if matched and (L, Ww) not in matched:
            continue

        set_dir = str(r.get("set_dir", "")).strip()
        if not set_dir or not os.path.isdir(set_dir):
            print(f"[WARN] invalid set_dir, skip: L={L} W={Ww} dir={set_dir}")
            continue

        eq_path = find_equity_file(set_dir, preferred=args.equity_filename)
        if not eq_path:
            print(f"[WARN] equity file not found in: {set_dir}")
            continue

        df = pd.read_csv(eq_path, encoding="utf-8-sig")
        cols = {c.lower(): c for c in df.columns}
        if "date" not in cols:
            print(f"[WARN] missing date column: {eq_path}"); continue
        dt_col = cols["date"]

        if "ret" in cols:
            rt = df[[dt_col, cols["ret"]]].copy()
            rt.rename(columns={cols["ret"]: "ret"}, inplace=True)
        elif "equity" in cols:
            rt = df[[dt_col, cols["equity"]]].copy()
            rt.rename(columns={cols["equity"]: "equity"}, inplace=True)
            rt["ret"] = rt["equity"].pct_change()
        else:
            print(f"[WARN] neither 'ret' nor 'equity' in {eq_path}"); continue

        rt[dt_col] = pd.to_datetime(rt[dt_col])
        rt = rt.dropna(subset=["ret"]).sort_values(dt_col)

        fam = f"{w2d.get(Ww,'?')}↔{Ww}"
        for _, x in rt.iterrows():
            out_rows.append(dict(
                date=x[dt_col],
                L=float(L),
                weekly_z=int(Ww),
                ret=float(x["ret"]),
                family=fam,
                source="weekly",
                set_dir=set_dir
            ))

    if not out_rows:
        print("[ERROR] no rows produced"); sys.exit(2)

    out_df = pd.DataFrame(out_rows).sort_values("date").reset_index(drop=True)
    default_out = os.path.join(os.path.dirname(os.path.abspath(args.best_sets_csv)), "ts_weekly.csv")
    out_path = args.out_csv or default_out
    out_df.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path}")
    print("Done.")

if __name__ == "__main__":
    main()