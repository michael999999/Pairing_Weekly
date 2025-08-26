#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thesis Figures Generator
- 從 gridsearch 輸出生成論文用圖表：
  1) L×Z Sharpe / MDD 熱力圖（best_sets.csv）
  2) 年度收益柱狀（最佳 Set）vs 基準（_GSPC 買進持有）
  3) 全期累積淨值曲線（最佳 Set）vs 基準

註解：繁體中文；print/log 英文
"""

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_best_sets(best_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(best_csv, encoding="utf-8-sig")
    # 型別正規化
    df["formation_length"] = pd.to_numeric(df["formation_length"], errors="coerce")
    df["z_window"] = pd.to_numeric(df["z_window"], errors="coerce").astype("Int64")
    df["sharpe"] = pd.to_numeric(df["sharpe"], errors="coerce")
    df["max_drawdown"] = pd.to_numeric(df["max_drawdown"], errors="coerce")
    return df


def plot_heatmaps(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    # Sharpe Pivot
    sharpe_pivot = (df.pivot(index="formation_length", columns="z_window", values="sharpe")
                      .sort_index().reindex(sorted(df["z_window"].dropna().unique()), axis=1))
    sharpe_pivot.to_csv(out_dir / "pivot_sharpe_best.csv", encoding="utf-8-sig")
    # MDD Pivot
    mdd_pivot = (df.pivot(index="formation_length", columns="z_window", values="max_drawdown")
                   .sort_index().reindex(sorted(df["z_window"].dropna().unique()), axis=1))
    mdd_pivot.to_csv(out_dir / "pivot_mdd_best.csv", encoding="utf-8-sig")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Sharpe
        plt.figure(figsize=(1.0 + 0.9 * sharpe_pivot.shape[1], 1.2 + 0.6 * sharpe_pivot.shape[0]))
        ax = sns.heatmap(sharpe_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
        ax.set_xlabel("z_window (weeks)")
        ax.set_ylabel("formation_length (years)")
        ax.set_title("Best Set Sharpe (by L×Z)")
        plt.tight_layout()
        plt.savefig(out_dir / "pivot_sharpe_best.png", dpi=160)
        plt.close()

        # MDD
        plt.figure(figsize=(1.0 + 0.9 * mdd_pivot.shape[1], 1.2 + 0.6 * mdd_pivot.shape[0]))
        ax = sns.heatmap(mdd_pivot, annot=True, fmt=".2%", cmap="PuRd")
        ax.set_xlabel("z_window (weeks)")
        ax.set_ylabel("formation_length (years)")
        ax.set_title("Best Set Max Drawdown (by L×Z)")
        plt.tight_layout()
        plt.savefig(out_dir / "pivot_mdd_best.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"[WARN] Plot heatmaps skipped: {e}")


def pick_best_overall(df: pd.DataFrame) -> pd.Series:
    """選擇 Sharpe 最佳的 Set；平手取 cum_return 高者，再取 ann_vol 低者。"""
    df2 = df.copy()
    df2["cum_return"] = pd.to_numeric(df2["cum_return"], errors="coerce")
    df2["ann_vol"] = pd.to_numeric(df2["ann_vol"], errors="coerce")
    s = df2.sort_values(["sharpe","cum_return","ann_vol"], ascending=[False,False,True]).iloc[0]
    return s


def load_benchmark_yearly(prices_path: Path, symbol: str, years: List[str]) -> pd.DataFrame:
    """從日價計算基準（買進持有）年度報酬。"""
    px = pd.read_pickle(prices_path)
    if symbol not in px.columns:
        raise KeyError(f"Benchmark symbol not found in prices: {symbol}")
    s = px[symbol].dropna()
    s.index = pd.to_datetime(s.index)
    # 逐年報酬：年末/年初 - 1
    res = []
    for y in years:
        y = str(y)
        s_y = s[(s.index >= f"{y}-01-01") & (s.index <= f"{y}-12-31")]
        if len(s_y) < 2:
            ret = np.nan
        else:
            ret = float(s_y.iloc[-1] / s_y.iloc[0] - 1.0)
        res.append(dict(trading_period=y, bench_return=ret))
    return pd.DataFrame(res)


def plot_yearly_bars(best_row: pd.Series, set_dir: Path, prices_path: Path, bench_symbol: str, out_dir: Path):
    ensure_dir(out_dir)
    yr_path = set_dir / "yearly_metrics.csv"
    yr = pd.read_csv(yr_path, encoding="utf-8-sig")
    yr["trading_period"] = yr["trading_period"].astype(str)
    yr["ann_return"] = pd.to_numeric(yr["ann_return"], errors="coerce")
    years = yr["trading_period"].tolist()

    # 基準（買進持有）年度報酬
    bench_df = load_benchmark_yearly(prices_path, bench_symbol, years)
    df = yr.merge(bench_df, on="trading_period", how="left")

    try:
        import matplotlib.pyplot as plt
        x = np.arange(len(df))
        width = 0.38
        plt.figure(figsize=(max(6, 0.7*len(df)+2), 4.5))
        plt.bar(x - width/2, df["ann_return"].values, width=width, label="Strategy", color="#1f77b4")
        plt.bar(x + width/2, df["bench_return"].values, width=width, label=f"{bench_symbol} Buy&Hold", color="#ff7f0e")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.xticks(x, df["trading_period"].tolist(), rotation=0)
        plt.ylabel("Annual Return")
        plt.title(f"Yearly Returns: Best Set (L={best_row['formation_length']}, Z={int(best_row['z_window'])}) vs {bench_symbol}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "yearly_returns_vs_benchmark.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"[WARN] Yearly bars skipped: {e}")


def plot_equity_vs_benchmark(best_row: pd.Series, set_dir: Path, prices_path: Path, bench_symbol: str, out_dir: Path):
    ensure_dir(out_dir)
    eq_path = set_dir / "equity_curve_full.csv"
    eq = pd.read_csv(eq_path, parse_dates=["date"])
    eq = eq.sort_values("date")
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")

    # 基準累積曲線（與策略日期對齊）
    px = pd.read_pickle(prices_path)
    if bench_symbol not in px.columns:
        print(f"[WARN] Benchmark symbol not found in prices: {bench_symbol}. Skip equity comparison.")
        return
    s = px[bench_symbol].dropna()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(eq["date"]).ffill()
    bench_eq = (s / s.iloc[0]).values

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4.8))
        plt.plot(eq["date"], eq["equity"], label="Strategy", color="#1f77b4")
        plt.plot(eq["date"], bench_eq, label=f"{bench_symbol} Buy&Hold", color="#ff7f0e", alpha=0.8)
        plt.ylabel("Cumulative Equity (normalized)")
        plt.title(f"Cumulative Equity: Best Set (L={best_row['formation_length']}, Z={int(best_row['z_window'])}) vs {bench_symbol}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "equity_vs_benchmark.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"[WARN] Equity plot skipped: {e}")


def main():
    ap = argparse.ArgumentParser(description="Generate thesis-ready plots from gridsearch outputs.")
    ap.add_argument("--grid-root", type=str, default="reports/gridsearch_weekly", help="Gridsearch output root.")
    ap.add_argument("--best-csv", type=str, default=None, help="Path to _summary/best_sets.csv (optional).")
    ap.add_argument("--prices", type=str, default="data/prices.pkl", help="Daily adjusted close (wide DataFrame).")
    ap.add_argument("--benchmark-symbol", type=str, default="_GSPC", help="Benchmark symbol in prices.pkl (buy&hold).")
    ap.add_argument("--out-dir", type=str, default="reports/thesis_figs", help="Output figures directory.")
    args = ap.parse_args()

    grid_root = Path(args.grid_root)
    best_csv = Path(args.best_csv) if args.best_csv else (grid_root / "_summary" / "best_sets.csv")
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print(f"[INFO] Loading best sets: {best_csv}")
    best = load_best_sets(best_csv)

    # 1) 熱力圖（Sharpe / MDD）
    plot_heatmaps(best, out_dir)

    # 2) 選 Sharpe 最佳的 Set，畫年度柱狀與全期曲線 vs 基準
    best_row = pick_best_overall(best)
    set_dir = Path(best_row["set_dir"]) if "set_dir" in best_row else (grid_root / f"L{int(round(best_row['formation_length']*100)):03d}_Z{int(best_row['z_window']):03d}")
    try:
        plot_yearly_bars(best_row, set_dir, Path(args.prices), args.benchmark_symbol, out_dir)
        plot_equity_vs_benchmark(best_row, set_dir, Path(args.prices), args.benchmark_symbol, out_dir)
    except Exception as e:
        print(f"[WARN] Benchmark comparison skipped: {e}")

    print(f"[INFO] Figures saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()