#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thesis Figures Generator (enhanced)
- 從 gridsearch 輸出生成論文用圖表：
  1) L×Z Sharpe / MDD 熱力圖（best_sets.csv）
  2) 年度收益柱狀（最佳 Set）vs 基準（_GSPC 買進持有）
  3) 全期累積淨值曲線（最佳 Set）vs 基準
  4) 年度 Sharpe 條圖（最佳 Set）
  5) 年度 MDD 箱型圖（by formation_length 與 by z_window；跨年度分佈）
  6) 風險–報酬散佈（AnnVol vs AnnReturn），色碼 L 或 Z
  7) 滾動 Sharpe（252、126 日）策略 vs 基準
  8) 年度超額報酬（策略 − 基準）柱狀
  9) 日報酬分布直方/密度圖（策略 vs 基準）
  10) 累積回撤時間序列（策略 vs 基準）

註解：繁體中文；print/log 英文
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_best_sets(best_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(best_csv, encoding="utf-8-sig")
    # 型別正規化
    if "formation_length" in df.columns:
        df["formation_length"] = pd.to_numeric(df["formation_length"], errors="coerce")
    if "z_window" in df.columns:
        df["z_window"] = pd.to_numeric(df["z_window"], errors="coerce").astype("Int64")
    for c in ("sharpe","max_drawdown","ann_return","ann_vol","cum_return"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def plot_heatmaps(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    # Sharpe Pivot（最佳 Set 版）
    sharpe_pivot = (df.pivot(index="formation_length", columns="z_window", values="sharpe")
                      .sort_index().reindex(sorted(df["z_window"].dropna().unique()), axis=1))
    sharpe_pivot.to_csv(out_dir / "pivot_sharpe_best.csv", encoding="utf-8-sig")
    # MDD Pivot（最佳 Set 版）
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
    for c in ("cum_return","ann_vol"):
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    s = df2.sort_values(["sharpe","cum_return","ann_vol"], ascending=[False,False,True]).iloc[0]
    return s


def load_benchmark_yearly(prices_path: Path, symbol: str, years: List[str]) -> pd.DataFrame:
    """從日價計算基準（買進持有）年度報酬。"""
    px = pd.read_pickle(prices_path)
    if symbol not in px.columns:
        raise KeyError(f"Benchmark symbol not found in prices: {symbol}")
    s = px[symbol].dropna()
    s.index = pd.to_datetime(s.index)
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
    """年度報酬柱狀（策略 vs 基準）。"""
    ensure_dir(out_dir)
    yr_path = set_dir / "yearly_metrics.csv"
    yr = pd.read_csv(yr_path, encoding="utf-8-sig")
    yr["trading_period"] = yr["trading_period"].astype(str)
    yr["ann_return"] = pd.to_numeric(yr["ann_return"], errors="coerce")
    years = yr["trading_period"].tolist()

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


def plot_yearly_sharpe(best_row: pd.Series, set_dir: Path, out_dir: Path):
    """年度 Sharpe 條圖（最佳 Set）。"""
    ensure_dir(out_dir)
    yr_path = set_dir / "yearly_metrics.csv"
    yr = pd.read_csv(yr_path, encoding="utf-8-sig")
    yr["trading_period"] = yr["trading_period"].astype(str)
    yr["sharpe"] = pd.to_numeric(yr["sharpe"], errors="coerce")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(max(6, 0.7*len(yr)+2), 3.8))
        plt.bar(yr["trading_period"], yr["sharpe"], color="#2ca02c")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.ylabel("Sharpe")
        plt.title(f"Yearly Sharpe: Best Set (L={best_row['formation_length']}, Z={int(best_row['z_window'])})")
        plt.tight_layout()
        plt.savefig(out_dir / "yearly_sharpe_best.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"[WARN] Yearly Sharpe bar skipped: {e}")


def plot_equity_vs_benchmark(best_row: pd.Series, set_dir: Path, prices_path: Path, bench_symbol: str, out_dir: Path):
    """全期累積曲線對比基準。"""
    ensure_dir(out_dir)
    eq_path = set_dir / "equity_curve_full.csv"
    eq = pd.read_csv(eq_path, parse_dates=["date"])
    eq = eq.sort_values("date")
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")

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


def rolling_sharpe(ret: pd.Series, window: int = 252) -> pd.Series:
    """滾動 Sharpe（年化，日頻）。"""
    mu = ret.rolling(window).mean() * 252
    sd = ret.rolling(window).std(ddof=1) * np.sqrt(252)
    return mu / sd.replace(0.0, np.nan)


def plot_rolling_sharpe(best_row: pd.Series, set_dir: Path, prices_path: Path, bench_symbol: str, out_dir: Path):
    """策略與基準的滾動 Sharpe（252 與 126 日）。"""
    ensure_dir(out_dir)
    eq_path = set_dir / "equity_curve_full.csv"
    eq = pd.read_csv(eq_path, parse_dates=["date"]).sort_values("date")
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    ret_s = eq["equity"].pct_change().fillna(0.0)

    px = pd.read_pickle(prices_path)
    if bench_symbol not in px.columns:
        print(f"[WARN] Benchmark symbol not found in prices: {bench_symbol}. Skip rolling Sharpe.")
        return
    s = px[bench_symbol].dropna()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(eq["date"]).ffill()
    ret_b = s.pct_change().fillna(0.0)

    for win, tag in [(252, "252d"), (126, "126d")]:
        rs_s = rolling_sharpe(ret_s, window=win)
        rs_b = rolling_sharpe(ret_b, window=win)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 4.2))
            plt.plot(eq["date"], rs_s, label="Strategy", color="#1f77b4")
            plt.plot(eq["date"], rs_b, label=f"{bench_symbol}", color="#ff7f0e", alpha=0.8)
            plt.axhline(0, color="gray", linewidth=0.8)
            plt.ylabel(f"Rolling Sharpe ({tag})")
            plt.title(f"Rolling Sharpe ({tag}): Best Set (L={best_row['formation_length']}, Z={int(best_row['z_window'])})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"rolling_sharpe_{tag}.png", dpi=160)
            plt.close()
        except Exception as e:
            print(f"[WARN] Rolling Sharpe {tag} skipped: {e}")


def plot_excess_bars(best_row: pd.Series, set_dir: Path, prices_path: Path, bench_symbol: str, out_dir: Path):
    """年度超額報酬（策略 − 基準）。"""
    ensure_dir(out_dir)
    yr_path = set_dir / "yearly_metrics.csv"
    yr = pd.read_csv(yr_path, encoding="utf-8-sig")
    yr["trading_period"] = yr["trading_period"].astype(str)
    yr["ann_return"] = pd.to_numeric(yr["ann_return"], errors="coerce")
    years = yr["trading_period"].tolist()
    bench_df = load_benchmark_yearly(prices_path, bench_symbol, years)
    df = yr.merge(bench_df, on="trading_period", how="left")
    df["excess"] = df["ann_return"] - df["bench_return"]

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(max(6, 0.7*len(df)+2), 3.8))
        colors = ["#2ca02c" if x >= 0 else "#d62728" for x in df["excess"]]
        plt.bar(df["trading_period"], df["excess"], color=colors)
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.ylabel("Excess Return (Strategy - Benchmark)")
        plt.title(f"Yearly Excess Returns vs {bench_symbol}: Best Set (L={best_row['formation_length']}, Z={int(best_row['z_window'])})")
        plt.tight_layout()
        plt.savefig(out_dir / "yearly_excess_returns.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"[WARN] Excess bars skipped: {e}")


def plot_daily_ret_hist(best_row: pd.Series, set_dir: Path, prices_path: Path, bench_symbol: str, out_dir: Path):
    """日報酬分布（策略 vs 基準）。"""
    ensure_dir(out_dir)
    eq_path = set_dir / "equity_curve_full.csv"
    eq = pd.read_csv(eq_path, parse_dates=["date"]).sort_values("date")
    ret_s = eq["equity"].pct_change().dropna()

    px = pd.read_pickle(prices_path)
    if bench_symbol not in px.columns:
        print(f"[WARN] Benchmark symbol not found in prices: {bench_symbol}. Skip return histogram.")
        return
    s = px[bench_symbol].dropna()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(eq["date"]).ffill()
    ret_b = s.pct_change().dropna()

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(7.5, 4.2))
        sns.histplot(ret_s, bins=80, stat="density", color="#1f77b4", alpha=0.4, label="Strategy")
        sns.histplot(ret_b, bins=80, stat="density", color="#ff7f0e", alpha=0.4, label=f"{bench_symbol}")
        sns.kdeplot(ret_s, color="#1f77b4")
        sns.kdeplot(ret_b, color="#ff7f0e")
        plt.xlabel("Daily Return")
        plt.title("Distribution of Daily Returns (Strategy vs Benchmark)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "daily_return_distribution.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"[WARN] Return histogram skipped: {e}")


def collect_yearly_from_sets(best_df: pd.DataFrame) -> pd.DataFrame:
    """
    掃描 best_sets.csv 中每個 set_dir 的 yearly_metrics.csv，彙整年度層級的指標。
    回傳欄位：formation_length, z_window, trading_period, ann_return, ann_vol, sharpe, max_drawdown
    """
    rows = []
    for _, r in best_df.iterrows():
        set_dir = Path(r["set_dir"]) if "set_dir" in r and isinstance(r["set_dir"], str) else None
        if set_dir is None or not set_dir.exists():
            # 嘗試用推導路徑
            set_dir = (Path("reports/gridsearch_weekly") / f"L{int(round(float(r['formation_length'])*100)):03d}_Z{int(r['z_window']):03d}")
        yr_path = set_dir / "yearly_metrics.csv"
        if not yr_path.exists():
            continue
        try:
            df = pd.read_csv(yr_path, encoding="utf-8-sig")
            df["formation_length"] = float(r["formation_length"])
            df["z_window"] = int(r["z_window"])
            rows.append(df[["formation_length","z_window","trading_period","ann_return","ann_vol","sharpe","max_drawdown"]])
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, axis=0, ignore_index=True)
    out["trading_period"] = out["trading_period"].astype(str)
    for c in ("ann_return","ann_vol","sharpe","max_drawdown"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def plot_mdd_boxplots(yearly_df: pd.DataFrame, out_dir: Path):
    """年度 MDD 箱型圖：by formation_length 與 by z_window（跨年度彙總分布）。"""
    if yearly_df.empty:
        print("[WARN] Yearly dataframe empty; skip MDD boxplots.")
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # by L
        plt.figure(figsize=(max(6, 1.1*len(yearly_df["formation_length"].unique())+2), 4.2))
        sns.boxplot(x="formation_length", y="max_drawdown", data=yearly_df, color="#9ecae1")
        plt.xlabel("formation_length (years)")
        plt.ylabel("Max Drawdown (yearly)")
        plt.title("Yearly Max Drawdown Distribution by formation_length")
        plt.tight_layout()
        plt.savefig(out_dir / "mdd_box_by_L.png", dpi=160)
        plt.close()

        # by Z
        plt.figure(figsize=(max(6, 1.1*len(yearly_df["z_window"].unique())+2), 4.2))
        sns.boxplot(x="z_window", y="max_drawdown", data=yearly_df, color="#fdae6b")
        plt.xlabel("z_window (weeks)")
        plt.ylabel("Max Drawdown (yearly)")
        plt.title("Yearly Max Drawdown Distribution by z_window")
        plt.tight_layout()
        plt.savefig(out_dir / "mdd_box_by_Z.png", dpi=160)
        plt.close()

    except Exception as e:
        print(f"[WARN] MDD boxplots skipped: {e}")


def plot_risk_return_scatter(best_df: pd.DataFrame, out_dir: Path):
    """風險–報酬散佈（AnnVol vs AnnReturn），色碼 L 或 Z。"""
    ensure_dir(out_dir)
    df = best_df.copy()
    if not {"ann_return","ann_vol","formation_length","z_window","max_drawdown"}.issubset(df.columns):
        print("[WARN] Missing columns for risk-return scatter; skip.")
        return
    for c in ("ann_return","ann_vol","max_drawdown","formation_length","z_window"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # by Z (color)，大小表徵 |MDD|
        plt.figure(figsize=(7.8, 5.4))
        sc = plt.scatter(df["ann_vol"], df["ann_return"], c=df["z_window"], cmap="viridis",
                         s=(df["max_drawdown"].abs()*300).clip(30, 300), alpha=0.9, edgecolor="k", linewidths=0.4)
        plt.xlabel("AnnVol")
        plt.ylabel("AnnReturn")
        plt.title("Risk-Return Scatter (color by z_window, size by |MDD|)")
        cbar = plt.colorbar(sc)
        cbar.set_label("z_window (weeks)")
        for _, r in df.iterrows():
            plt.annotate(f"L{r['formation_length']}", (r["ann_vol"], r["ann_return"]), textcoords="offset points", xytext=(4,4), fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / "risk_return_scatter_byZ.png", dpi=160)
        plt.close()

        # by L (color)
        plt.figure(figsize=(7.8, 5.4))
        sc = plt.scatter(df["ann_vol"], df["ann_return"], c=df["formation_length"], cmap="plasma",
                         s=(df["max_drawdown"].abs()*300).clip(30, 300), alpha=0.9, edgecolor="k", linewidths=0.4)
        plt.xlabel("AnnVol")
        plt.ylabel("AnnReturn")
        plt.title("Risk-Return Scatter (color by formation_length, size by |MDD|)")
        cbar = plt.colorbar(sc)
        cbar.set_label("formation_length (years)")
        for _, r in df.iterrows():
            plt.annotate(f"Z{int(r['z_window'])}", (r["ann_vol"], r["ann_return"]), textcoords="offset points", xytext=(4,4), fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / "risk_return_scatter_byL.png", dpi=160)
        plt.close()

    except Exception as e:
        print(f"[WARN] Risk-return scatter skipped: {e}")


def plot_drawdown_ts(best_row: pd.Series, set_dir: Path, prices_path: Path, bench_symbol: str, out_dir: Path):
    """累積回撤時間序列（策略 vs 基準）。"""
    ensure_dir(out_dir)
    eq_path = set_dir / "equity_curve_full.csv"
    eq = pd.read_csv(eq_path, parse_dates=["date"]).sort_values("date")
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    # 策略回撤
    peak = eq["equity"].cummax()
    dd_s = eq["equity"]/peak - 1.0

    px = pd.read_pickle(prices_path)
    if bench_symbol not in px.columns:
        print(f"[WARN] Benchmark symbol not found in prices: {bench_symbol}. Skip drawdown TS.")
        return
    s = px[bench_symbol].dropna()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(eq["date"]).ffill()
    bench_eq = (s / s.iloc[0])
    peak_b = bench_eq.cummax()
    dd_b = bench_eq/peak_b - 1.0

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4.2))
        plt.plot(eq["date"], dd_s, label="Strategy", color="#1f77b4")
        plt.plot(eq["date"], dd_b, label=f"{bench_symbol}", color="#ff7f0e", alpha=0.8)
        plt.ylabel("Drawdown")
        plt.title(f"Drawdown Over Time: Best Set (L={best_row['formation_length']}, Z={int(best_row['z_window'])})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "drawdown_timeseries.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"[WARN] Drawdown TS skipped: {e}")


def main():
    ap = argparse.ArgumentParser(description="Generate thesis-ready plots from gridsearch outputs (enhanced).")
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
    if best.empty:
        print("[ERROR] best_sets.csv empty or not found.")
        return

    # 1) 熱力圖（Sharpe / MDD）
    plot_heatmaps(best, out_dir)

    # 2) 選 Sharpe 最佳的 Set，畫年度柱狀、Sharpe、全期曲線 vs 基準、滾動 Sharpe、超額報酬、日報酬分布、回撤 TS
    best_row = pick_best_overall(best)
    set_dir = Path(best_row["set_dir"]) if "set_dir" in best_row and isinstance(best_row["set_dir"], str) else (grid_root / f"L{int(round(best_row['formation_length']*100)):03d}_Z{int(best_row['z_window']):03d}")
    try:
        plot_yearly_bars(best_row, set_dir, Path(args.prices), args.benchmark_symbol, out_dir)
        plot_yearly_sharpe(best_row, set_dir, out_dir)
        plot_equity_vs_benchmark(best_row, set_dir, Path(args.prices), args.benchmark_symbol, out_dir)
        plot_rolling_sharpe(best_row, set_dir, Path(args.prices), args.benchmark_symbol, out_dir)
        plot_excess_bars(best_row, set_dir, Path(args.prices), args.benchmark_symbol, out_dir)
        plot_daily_ret_hist(best_row, set_dir, Path(args.prices), args.benchmark_symbol, out_dir)
        plot_drawdown_ts(best_row, set_dir, Path(args.prices), args.benchmark_symbol, out_dir)
    except Exception as e:
        print(f"[WARN] Best-set comparative plots skipped: {e}")

    # 3) 年度 MDD 箱型圖（跨 L 與 Z）
    yearly_df = collect_yearly_from_sets(best)
    plot_mdd_boxplots(yearly_df, out_dir)

    # 4) 風險–報酬散佈（AnnVol vs AnnReturn）
    plot_risk_return_scatter(best, out_dir)

    print(f"[INFO] Figures saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()