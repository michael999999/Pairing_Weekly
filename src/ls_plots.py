#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ls_plots.py
- 讀取 daily/weekly 兩份 long_short_summary.csv，比較 Long 與 Short 的年化報酬（以及 Total）
- 圖：
  1) bars_long_short_by_family.png：依家族（21↔4, 42↔8, ...）比較 Daily vs Weekly 的 Long/Short 年化報酬
  2) bars_long_short_total_by_family.png：依家族比較 Daily vs Weekly 的 Total 年化報酬
- 備註：family 由 daily 的 z_window 映射到 weekly 的 z 視窗；weekly 反向映射到 daily，確保相同字串（例如 "126↔26"）
註解：繁體中文；print/log：英文
"""
import os
import sys
import argparse
import logging
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    out = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        d,w = tok.split(":")
        out[int(float(d.strip()))] = int(float(w.strip()))
    return out

def setup_style(dpi=150):
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9

def fmt_pct(x: float, d: int=2, dash="—") -> str:
    try:
        if x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x))):
            return dash
        return f"{float(x)*100:.{d}f}%"
    except Exception:
        return dash

def main():
    ap = argparse.ArgumentParser(description="Plot long/short decomposition (Daily vs Weekly).")
    ap.add_argument("--daily-summary", type=str, required=True, help="Path to daily long_short_summary.csv")
    ap.add_argument("--weekly-summary", type=str, required=True, help="Path to weekly long_short_summary.csv")
    ap.add_argument("--mapping", type=str, default="21:4,42:8,63:13,126:26,252:52", help="Daily→Weekly mapping for family tag.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for plots.")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = ap.parse_args()

    setup_logger("INFO")
    setup_style(args.dpi)
    ensure_dir(args.out_dir)

    d = pd.read_csv(args.daily_summary)
    w = pd.read_csv(args.weekly_summary)

    # 家族標籤
    mapping = parse_mapping(args.mapping)         # daily→weekly
    rev = {v:k for k,v in mapping.items()}        # weekly→daily

    d["family"] = d["z_window"].astype(int).map(lambda z: f"{int(z)}↔{mapping.get(int(z),'?')}")
    w["family"] = w["z_window"].astype(int).map(lambda z: f"{rev.get(int(z),'?')}↔{int(z)}")

    # 取交集家族
    fams = sorted(set(d["family"]).intersection(set(w["family"])))

    if not fams:
        print("[ERROR] No common families between daily and weekly."); sys.exit(2)

    # 只保留交集，並以 family 作 group（如同一 family 有多個 L，可先平均或保留所有列）
    dd = d[d["family"].isin(fams)].copy()
    ww = w[w["family"].isin(fams)].copy()

    # 先按 family 平均（如需 per-L 比較可自行調整）
    dG = dd.groupby("family").agg(
        ann_return_long=("ann_return_long","mean"),
        ann_return_short=("ann_return_short","mean"),
        ann_return_total=("ann_return_total","mean")
    ).reset_index()
    wG = ww.groupby("family").agg(
        ann_return_long=("ann_return_long","mean"),
        ann_return_short=("ann_return_short","mean"),
        ann_return_total=("ann_return_total","mean")
    ).reset_index()

    # 合併
    M = pd.merge(dG, wG, on="family", suffixes=("_daily","_weekly"))

    # 長/短邊年化報酬比較（%）
    labels = M["family"].tolist()
    x = np.arange(len(labels)); Wbar=0.18

    plt.figure(figsize=(max(8, 0.6*len(labels)+2), 4.6))
    plt.bar(x - 1.5*Wbar, M["ann_return_long_daily"]*100.0,  width=Wbar, label="Daily-Long",  color="tab:blue", alpha=0.75)
    plt.bar(x - 0.5*Wbar, M["ann_return_short_daily"]*100.0, width=Wbar, label="Daily-Short", color="tab:orange", alpha=0.75)
    plt.bar(x + 0.5*Wbar, M["ann_return_long_weekly"]*100.0, width=Wbar, label="Weekly-Long",  color="tab:green", alpha=0.75)
    plt.bar(x + 1.5*Wbar, M["ann_return_short_weekly"]*100.0,width=Wbar, label="Weekly-Short", color="tab:red", alpha=0.75)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Annualized return (%)")
    plt.title("Long/Short annualized return by family (Daily vs Weekly)")
    plt.legend(ncol=2)
    plt.tight_layout()
    fp = os.path.join(args.out_dir, "bars_long_short_by_family.png")
    plt.savefig(fp); plt.close()
    print(f"[WRITE] {fp}")

    # Total 年化報酬比較（%）
    plt.figure(figsize=(max(8, 0.6*len(labels)+2), 4.6))
    plt.bar(x - Wbar/2, M["ann_return_total_daily"]*100.0,  width=0.9*Wbar, label="Daily-Total",  color="tab:purple", alpha=0.80)
    plt.bar(x + Wbar/2, M["ann_return_total_weekly"]*100.0, width=0.9*Wbar, label="Weekly-Total", color="tab:brown",  alpha=0.80)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Annualized return (%)")
    plt.title("Total annualized return by family (Daily vs Weekly)")
    plt.legend()
    plt.tight_layout()
    fp = os.path.join(args.out_dir, "bars_long_short_total_by_family.png")
    plt.savefig(fp); plt.close()
    print(f"[WRITE] {fp}")

    print("Done.")

if __name__ == "__main__":
    main()