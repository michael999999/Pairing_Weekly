#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wf_global_plots.py
- 從 wf_yearly_global_pick 之輸出生成圖表：
  1) 年度收益柱狀（策略 vs 基準）
  2) 年度 Sharpe 柱狀（策略 vs 基準）
  3) 年度 Max Drawdown 柱狀（策略 vs 基準）
  4) 全期間累積淨值（策略 vs 基準；x 軸只顯示年份）

註解：繁體中文；print/log 英文
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -------- 基準處理 --------

def load_benchmark_series(prices_path: Path, symbol: str, dates: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    """載入基準等權（買進持有），回傳 (bench_equity_norm, bench_daily_ret) 對齊至指定 dates。"""
    px = pd.read_pickle(prices_path)
    if symbol not in px.columns:
        raise KeyError(f"Benchmark symbol not found: {symbol}")
    s = px[symbol].dropna()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(dates).ffill()
    eq = s / s.iloc[0]
    ret = s.pct_change().fillna(0.0)
    return eq, ret


def bench_yearly_from_prices(prices_path: Path, symbol: str, years: List[str]) -> pd.DataFrame:
    """基準年度指標：年報酬、年 Sharpe、年 MaxDD（以日頻年化 252）。"""
    px = pd.read_pickle(prices_path)
    if symbol not in px.columns:
        raise KeyError(f"Benchmark symbol not found: {symbol}")
    s = px[symbol].dropna()
    s.index = pd.to_datetime(s.index)

    rows = []
    for y in years:
        y = str(y)
        s_y = s[(s.index >= f"{y}-01-01") & (s.index <= f"{y}-12-31")]
        if len(s_y) < 3:
            rows.append(dict(year=y, bench_return=np.nan, bench_sharpe=np.nan, bench_mdd=np.nan))
            continue
        # 年報酬
        ret_y = float(s_y.iloc[-1] / s_y.iloc[0] - 1.0)
        # 年 Sharpe（以日頻年化）
        r_daily = s_y.pct_change().dropna()
        mu = r_daily.mean() * 252
        vol = r_daily.std(ddof=1) * np.sqrt(252) if r_daily.std(ddof=1) > 0 else np.nan
        sharpe = float(mu / vol) if vol and vol == vol else np.nan
        # 年度 MaxDD
        eq = s_y / s_y.iloc[0]
        peak = eq.cummax()
        mdd = float((eq / peak - 1.0).min())
        rows.append(dict(year=y, bench_return=ret_y, bench_sharpe=sharpe, bench_mdd=mdd))
    return pd.DataFrame(rows)


# -------- 畫圖 --------

def main():
    ap = argparse.ArgumentParser(description="Generate figures for WF yearly global best vs benchmark.")
    ap.add_argument("--wf-root", type=str, default="reports/wf_yearly_global_pick", help="WF global-pick output root.")
    ap.add_argument("--prices", type=str, default="data/prices.pkl", help="Daily adjusted close wide DataFrame.")
    ap.add_argument("--benchmark-symbol", type=str, default="_GSPC", help="Benchmark symbol in prices.pkl.")
    ap.add_argument("--out-dir", type=str, default="reports/wf_yearly_global_figs", help="Output dir for figures.")
    args = ap.parse_args()

    wf_root = Path(args.wf_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    yearly_csv = wf_root / "global_reopt_oos_yearly.csv"
    returns_csv = wf_root / "global_reopt_oos_returns.csv"
    if not yearly_csv.exists() or not returns_csv.exists():
        print(f"[ERROR] Missing WF outputs: {yearly_csv} or {returns_csv}")
        return

    # 讀逐年 OOS 指標（策略）
    ydf = pd.read_csv(yearly_csv, encoding="utf-8-sig")
    ydf["year"] = ydf["year"].astype(str)
    for c in ("ann_return","ann_vol","sharpe","max_drawdown"):
        if c in ydf.columns:
            ydf[c] = pd.to_numeric(ydf[c], errors="coerce")

    years = ydf["year"].tolist()

    # 基準年度指標
    bdf = bench_yearly_from_prices(Path(args.prices), args.benchmark_symbol, years)

    # 逐年合併（收益、Sharpe、MDD）
    df_year = ydf.merge(bdf, left_on="year", right_on="year", how="left")

    # 讀全期累積與日報酬
    rdf = pd.read_csv(returns_csv, parse_dates=["date"])
    rdf = rdf.sort_values("date")
    dates = pd.to_datetime(rdf["date"])
    rdf["equity"] = pd.to_numeric(rdf["equity"], errors="coerce")
    rdf["ret"] = pd.to_numeric(rdf["ret"], errors="coerce").fillna(0.0)

    # 基準全期序列
    bench_eq, bench_ret = load_benchmark_series(Path(args.prices), args.benchmark_symbol, dates)

    # 螢幕列印全期比較
    def ann_metrics(ret: pd.Series, freq: int = 252) -> dict:
        r = ret.dropna()
        if len(r) == 0:
            return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0)
        mu = r.mean() * freq
        vol = r.std(ddof=1) * np.sqrt(freq) if r.std(ddof=1) > 0 else 0.0
        sharpe = mu / vol if vol > 0 else 0.0
        return dict(ann_return=float(mu), ann_vol=float(vol), sharpe=float(sharpe))

    def max_dd(eq: pd.Series) -> float:
        peak = eq.cummax()
        return float((eq / peak - 1.0).min()) if len(eq) else np.nan

    m_s = ann_metrics(rdf["ret"])
    m_b = ann_metrics(bench_ret)
    m_s["max_drawdown"] = max_dd(rdf["equity"])
    m_b["max_drawdown"] = max_dd(bench_eq)

    print("Full-period metrics (Strategy vs Benchmark):")
    print(f"Strategy: Sharpe={m_s['sharpe']:.2f} AnnRet={m_s['ann_return']:.2%} AnnVol={m_s['ann_vol']:.2%} MDD={m_s['max_drawdown']:.2%}")
    print(f"Benchmark({args.benchmark_symbol}): Sharpe={m_b['sharpe']:.2f} AnnRet={m_b['ann_return']:.2%} AnnVol={m_b['ann_vol']:.2%} MDD={m_b['max_drawdown']:.2%}")

    # 繪圖
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.dates as mdates

        # 年度收益柱狀
        plt.figure(figsize=(max(6, 0.7*len(df_year)+2), 4.2))
        x = np.arange(len(df_year))
        width = 0.38
        plt.bar(x - width/2, df_year["ann_return"].values, width=width, label="Strategy", color="#1f77b4")
        plt.bar(x + width/2, df_year["bench_return"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.xticks(x, df_year["year"].tolist(), rotation=0)
        plt.ylabel("Annual Return")
        plt.title("Yearly Returns (OOS)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "yearly_returns_vs_benchmark.png", dpi=160)
        plt.close()

        # 年度 Sharpe 柱狀
        plt.figure(figsize=(max(6, 0.7*len(df_year)+2), 4.2))
        plt.bar(x - width/2, df_year["sharpe"].values, width=width, label="Strategy", color="#2ca02c")
        plt.bar(x + width/2, df_year["bench_sharpe"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.xticks(x, df_year["year"].tolist(), rotation=0)
        plt.ylabel("Sharpe")
        plt.title("Yearly Sharpe (OOS)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "yearly_sharpe_vs_benchmark.png", dpi=160)
        plt.close()

        # 年度 MDD 柱狀（值為負，越小越差）
        plt.figure(figsize=(max(6, 0.7*len(df_year)+2), 4.2))
        plt.bar(x - width/2, df_year["max_drawdown"].values, width=width, label="Strategy", color="#1f77b4")
        plt.bar(x + width/2, df_year["bench_mdd"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
        plt.xticks(x, df_year["year"].tolist(), rotation=0)
        plt.ylabel("Max Drawdown")
        plt.title("Yearly Max Drawdown (OOS)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "yearly_mdd_vs_benchmark.png", dpi=160)
        plt.close()

        # 全期間累積淨值（x 軸顯示年份）
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        ax.plot(dates, rdf["equity"].values, label="Strategy", color="#1f77b4")
        ax.plot(dates, bench_eq.values, label=f"{args.benchmark_symbol}", color="#ff7f0e", alpha=0.85)
        ax.set_ylabel("Cumulative Equity (norm.)")
        ax.set_title("Cumulative Equity (OOS)")
        ax.legend()
        # x 軸只顯示年份
        year_locator = mdates.YearLocator(base=1)
        year_fmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_locator(year_locator)
        ax.xaxis.set_major_formatter(year_fmt)
        ax.set_xlim(dates.min(), dates.max())
        plt.tight_layout()
        plt.savefig(out_dir / "cumulative_equity_vs_benchmark.png", dpi=160)
        plt.close()

        # 另外輸出合併的年度指標表（CSV）
        df_year.to_csv(out_dir / "yearly_strategy_vs_benchmark.csv", index=False, encoding="utf-8-sig")

        print(f"[INFO] Figures saved under: {out_dir.resolve()}")

    except Exception as e:
        print(f"[WARN] Plotting skipped: {e}")
        # 仍輸出年度 CSV
        df_year.to_csv(out_dir / "yearly_strategy_vs_benchmark.csv", index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved yearly CSV only: { (out_dir / 'yearly_strategy_vs_benchmark.csv').resolve() }")


if __name__ == "__main__":
    main()