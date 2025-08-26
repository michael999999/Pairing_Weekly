#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Walk-forward picks figure wrapper
- 讀取 wf_final_picks_by_W.csv（每個 W 的最終 L）
- 對每個 pick 生成一張綜合圖（年度柱狀、累積淨值、滾動 Sharpe、回撤），皆與基準 (_GSPC) 對比
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_bench_series(prices_path: Path, symbol: str, dates: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    """載入基準，對齊至策略日期，回傳 (bench_equity_norm, bench_daily_ret)。"""
    px = pd.read_pickle(prices_path)
    if symbol not in px.columns:
        raise KeyError(f"Benchmark symbol not found in prices: {symbol}")
    s = px[symbol].dropna()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(dates).ffill()
    eq = s / s.iloc[0]
    ret = s.pct_change().fillna(0.0)
    return eq, ret


def load_bench_yearly(prices_path: Path, symbol: str, years: List[str]) -> pd.DataFrame:
    """計算基準年度報酬（年末/年初 − 1）。"""
    px = pd.read_pickle(prices_path)
    if symbol not in px.columns:
        raise KeyError(f"Benchmark symbol not found in prices: {symbol}")
    s = px[symbol].dropna()
    s.index = pd.to_datetime(s.index)
    rows = []
    for y in years:
        sy = s[(s.index >= f"{y}-01-01") & (s.index <= f"{y}-12-31")]
        if len(sy) < 2:
            r = np.nan
        else:
            r = float(sy.iloc[-1] / sy.iloc[0] - 1.0)
        rows.append(dict(trading_period=y, bench_return=r))
    return pd.DataFrame(rows)


def rolling_sharpe(ret: pd.Series, window: int = 252) -> pd.Series:
    mu = ret.rolling(window).mean() * 252
    sd = ret.rolling(window).std(ddof=1) * np.sqrt(252)
    return mu / sd.replace(0.0, np.nan)


def plot_one_pick(pick_row: pd.Series, prices_path: Path, bench_symbol: str, out_dir: Path):
    """生成單一 pick 的四合一綜合圖。"""
    set_dir = Path(pick_row["set_dir"])
    ensure_dir(out_dir)

    # 載入 OOS 日報酬/曲線
    wf_ret_path = set_dir / "wf_oos_returns.csv"
    if not wf_ret_path.exists():
        print(f"[WARN] Missing OOS returns: {wf_ret_path}")
        return
    oos = pd.read_csv(wf_ret_path, parse_dates=["date"]).sort_values("date")
    oos["ret"] = pd.to_numeric(oos["ret"], errors="coerce").fillna(0.0)
    oos["equity"] = pd.to_numeric(oos["equity"], errors="coerce")
    dates = pd.to_datetime(oos["date"])

    # 基準
    bench_eq, bench_ret = load_bench_series(prices_path, bench_symbol, dates)

    # 年度表
    yr_path = set_dir / "wf_oos_params.csv"
    yr = pd.read_csv(yr_path) if yr_path.exists() else pd.DataFrame()
    if not yr.empty:
        yr["year"] = yr["year"].astype(str)
        yr["ann_return"] = pd.to_numeric(yr["ann_return"], errors="coerce")
        years = yr["year"].tolist()
        bench_year = load_bench_yearly(prices_path, bench_symbol, years)
        yr = yr.merge(bench_year.rename(columns={"trading_period": "year"}), on="year", how="left")
    else:
        years = []
        yr = pd.DataFrame(columns=["year","ann_return","bench_return"])

    # 圖
    try:
        import matplotlib.pyplot as plt

        L = pick_row["formation_length"]; Z = int(pick_row["z_window"])
        title = f"WF Pick L={L}, Z={Z} ({bench_symbol} as benchmark)"

        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
        ax1, ax2, ax3, ax4 = axes.flatten()

        # 年度柱狀
        if not yr.empty:
            x = np.arange(len(yr))
            width = 0.38
            ax1.bar(x - width/2, yr["ann_return"].values, width=width, label="Strategy", color="#1f77b4")
            ax1.bar(x + width/2, yr["bench_return"].values, width=width, label=f"{bench_symbol}", color="#ff7f0e")
            ax1.axhline(0, color="gray", linewidth=0.8)
            ax1.set_xticks(x); ax1.set_xticklabels(yr["year"].tolist(), rotation=0)
            ax1.set_ylabel("Annual Return")
            ax1.set_title("Yearly Returns (OOS)")
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, "No yearly data", ha="center", va="center"); ax1.axis("off")

        # 累積淨值
        ax2.plot(dates, oos["equity"], label="Strategy", color="#1f77b4")
        ax2.plot(dates, bench_eq.values, label=f"{bench_symbol}", color="#ff7f0e", alpha=0.85)
        ax2.set_ylabel("Cumulative Equity (norm.)")
        ax2.set_title("Cumulative Equity (OOS)")
        ax2.legend()

        # 滾動 Sharpe（252）
        rs_s = rolling_sharpe(oos["ret"], window=252)
        rs_b = rolling_sharpe(bench_ret, window=252)
        ax3.plot(dates, rs_s, label="Strategy", color="#1f77b4")
        ax3.plot(dates, rs_b, label=f"{bench_symbol}", color="#ff7f0e", alpha=0.85)
        ax3.axhline(0, color="gray", linewidth=0.8)
        ax3.set_ylabel("Rolling Sharpe (252d)")
        ax3.set_title("Rolling Sharpe (OOS)")
        ax3.legend()

        # 回撤
        eq_s = oos["equity"].astype(float)
        peak_s = eq_s.cummax(); dd_s = eq_s/peak_s - 1.0
        peak_b = bench_eq.cummax(); dd_b = bench_eq/peak_b - 1.0
        ax4.plot(dates, dd_s.values, label="Strategy", color="#1f77b4")
        ax4.plot(dates, dd_b.values, label=f"{bench_symbol}", color="#ff7f0e", alpha=0.85)
        ax4.set_ylabel("Drawdown")
        ax4.set_title("Drawdown (OOS)")
        ax4.legend()

        fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = out_dir / f"wf_pick_L{int(round(float(L)*100)):03d}_Z{Z:03d}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"[INFO] Saved figure: {out_path.resolve()}")
    except Exception as e:
        print(f"[WARN] Plot failed for pick: {e}")


def main():
    ap = argparse.ArgumentParser(description="Plot composite figures for WF picks (per W final L).")
    ap.add_argument("--wf-root", type=str, default="reports/walkforward_weekly", help="Walk-forward output root.")
    ap.add_argument("--prices", type=str, default="data/prices.pkl", help="Daily adjusted close path.")
    ap.add_argument("--benchmark-symbol", type=str, default="_GSPC", help="Benchmark symbol in prices.pkl.")
    ap.add_argument("--out-dir", type=str, default="reports/walkforward_figs", help="Output dir for pick figures.")
    args = ap.parse_args()

    wf_root = Path(args.wf_root)
    picks_csv = wf_root / "_summary" / "wf_final_picks_by_W.csv"
    if not picks_csv.exists():
        print(f"[ERROR] Picks CSV not found: {picks_csv}")
        return
    picks = pd.read_csv(picks_csv, encoding="utf-8-sig")
    if picks.empty:
        print("[ERROR] Picks CSV is empty.")
        return

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    for _, row in picks.iterrows():
        plot_one_pick(row, Path(args.prices), args.benchmark_symbol, out_dir)


if __name__ == "__main__":
    main()