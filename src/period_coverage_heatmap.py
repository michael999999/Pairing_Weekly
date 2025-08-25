#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
列出每個年度 × formation_length 的唯一配對（方向化 pair_id）計數透視表，並輸出熱力圖。
- 註解：繁體中文
- print/log：英文
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Build coverage heatmap (year × formation_length) of unique pairs.")
    ap.add_argument("--pairs", type=str, default="cache/top_pairs_annual.csv", help="Path to selection CSV.")
    ap.add_argument("--out-csv", type=str, default="reports/coverage_counts.csv", help="Path to output pivot CSV.")
    ap.add_argument("--out-png", type=str, default="reports/coverage_heatmap.png", help="Path to output heatmap PNG.")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    args = ap.parse_args()

    print(f"[INFO] Loading: {args.pairs}")
    df = pd.read_csv(args.pairs, encoding="utf-8-sig", low_memory=False)

    # 方向化 pair_id（若未存在則組合）
    if "pair_id" not in df.columns:
        if "stock1" in df.columns and "stock2" in df.columns:
            df["pair_id"] = df["stock1"].astype(str) + "__" + df["stock2"].astype(str)
        elif "pair" in df.columns:
            df["pair_id"] = df["pair"].astype(str)
        else:
            raise KeyError("Need 'pair_id' or ('stock1','stock2') in CSV.")

    # 型別正規化
    df["trading_period_str"] = df["trading_period"].astype(str)
    df["formation_length_float"] = pd.to_numeric(df["formation_length"], errors="coerce")

    # 以唯一 pair_id 計數（避免重覆列）
    # 若同一年度同一 L 出現重覆 pair_id，只算一次
    grp = df.dropna(subset=["formation_length_float"]).groupby(
        ["trading_period_str", "formation_length_float"]
    )["pair_id"].nunique()

    # 透視表：index=年度（字串），columns=formation_length（數值）
    pivot = grp.unstack(fill_value=0)

    # 排序欄列
    pivot = pivot.sort_index()  # 年度排序
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)  # L 由小到大

    # 輸出 CSV
    out_csv = Path(args.out_csv)
    ensure_dir(out_csv)
    pivot.to_csv(out_csv, encoding="utf-8-sig")
    print(f"[INFO] Saved pivot CSV: {out_csv.resolve()}")

    # 繪圖（可選）
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig_h = max(4, min(0.3 * len(pivot.index) + 2, 18))
        fig_w = max(6, min(0.5 * len(pivot.columns) + 2, 18))
        plt.figure(figsize=(fig_w, fig_h))
        ax = sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
        ax.set_xlabel("formation_length (years)")
        ax.set_ylabel("trading_period (year)")
        ax.set_title("Unique pairs count by Year × Formation Length")
        plt.tight_layout()

        out_png = Path(args.out_png)
        ensure_dir(out_png)
        plt.savefig(out_png, dpi=args.dpi)
        plt.close()
        print(f"[INFO] Saved heatmap PNG: {out_png.resolve()}")
    except Exception as e:
        print(f"[WARN] Plot skipped (missing seaborn/matplotlib?): {e}")

if __name__ == "__main__":
    main()