#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wf_global_plots.py (with target-vol option)
- 從 wf_yearly_global_pick 之輸出生成圖表：
  1) 年度收益柱狀（策略 vs 基準）
  2) 年度 Sharpe 柱狀（策略 vs 基準）
  3) 年度 Max Drawdown 柱狀（策略 vs 基準）
  4) 全期間累積淨值（策略 vs 基準；x 軸只顯示年份）
- 新增：--target-vol（若 >0，將策略日報酬以常數槓桿縮放至目標年化波動，另輸出對應圖與指標）
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_benchmark_series(prices_path: Path, symbol: str, dates: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    px = pd.read_pickle(prices_path)
    if symbol not in px.columns:
        raise KeyError(f"Benchmark symbol not found: {symbol}")
    s = px[symbol].dropna(); s.index = pd.to_datetime(s.index)
    s = s.reindex(dates).ffill()
    eq = s / s.iloc[0]
    ret = s.pct_change().fillna(0.0)
    return eq, ret

def bench_yearly(prices_path: Path, symbol: str, years: List[str]) -> pd.DataFrame:
    px = pd.read_pickle(prices_path)
    if symbol not in px.columns:
        raise KeyError(f"Benchmark symbol not found: {symbol}")
    s = px[symbol].dropna(); s.index = pd.to_datetime(s.index)
    rows=[]
    for y in years:
        sy = s[(s.index >= f"{y}-01-01") & (s.index <= f"{y}-12-31")]
        if len(sy)<3:
            rows.append(dict(year=str(y), bench_return=np.nan, bench_sharpe=np.nan, bench_mdd=np.nan))
            continue
        ret_y = float(sy.iloc[-1] / sy.iloc[0] - 1.0)
        r_d = sy.pct_change().dropna()
        mu = r_d.mean()*252; vol = r_d.std(ddof=1)*np.sqrt(252) if r_d.std(ddof=1)>0 else np.nan
        sharpe = float(mu/vol) if vol==vol and vol else np.nan
        eq = sy/sy.iloc[0]; mdd = float((eq/eq.cummax()-1.0).min())
        rows.append(dict(year=str(y), bench_return=ret_y, bench_sharpe=sharpe, bench_mdd=mdd))
    return pd.DataFrame(rows)

def ann_metrics(ret: pd.Series, freq: int=252) -> dict:
    r=ret.dropna()
    if len(r)==0:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0)
    mu=r.mean()*freq; vol=r.std(ddof=1)*np.sqrt(freq) if r.std(ddof=1)>0 else 0.0
    sharpe=mu/vol if vol>0 else 0.0
    return dict(ann_return=float(mu), ann_vol=float(vol), sharpe=float(sharpe))

def max_dd(eq: pd.Series) -> float:
    if len(eq)==0: return np.nan
    peak=eq.cummax(); return float((eq/peak-1.0).min())

def to_yearly(ret: pd.Series) -> pd.DataFrame:
    """將日報酬聚合為年度報酬（複利）。"""
    df = ret.to_frame("ret")
    df["year"] = df.index.year
    out = df.groupby("year").apply(lambda g: (1.0+g["ret"]).prod()-1.0).rename("ret").reset_index()
    out["year"] = out["year"].astype(str)
    return out

def main():
    ap = argparse.ArgumentParser(description="WF global-pick plots vs benchmark (with target-vol option).")
    ap.add_argument("--wf-root", type=str, default="reports/wf_yearly_global_pick")
    ap.add_argument("--prices", type=str, default="data/prices.pkl")
    ap.add_argument("--benchmark-symbol", type=str, default="_GSPC")
    ap.add_argument("--out-dir", type=str, default="reports/wf_yearly_global_figs")
    ap.add_argument("--target-vol", type=float, default=0.0, help="If >0, target annualized vol (e.g., 10 for 10%).")
    args = ap.parse_args()

    wf_root = Path(args.wf_root); out_dir = Path(args.out_dir); ensure_dir(out_dir)
    yearly_csv  = wf_root / "global_reopt_oos_yearly.csv"
    returns_csv = wf_root / "global_reopt_oos_returns.csv"
    if not yearly_csv.exists() or not returns_csv.exists():
        print(f"[ERROR] Missing WF outputs: {yearly_csv} or {returns_csv}")
        return

    ydf = pd.read_csv(yearly_csv, encoding="utf-8-sig")
    ydf["year"] = ydf["year"].astype(str)
    for c in ("ann_return","ann_vol","sharpe","max_drawdown"):
        if c in ydf.columns: ydf[c]=pd.to_numeric(ydf[c], errors="coerce")

    rdf = pd.read_csv(returns_csv, parse_dates=["date"]).sort_values("date")
    # 轉型並設置日期為索引（DatetimeIndex）
    rdf["ret"]    = pd.to_numeric(rdf["ret"], errors="coerce").fillna(0.0)
    rdf["equity"] = pd.to_numeric(rdf["equity"], errors="coerce")
    rdf = rdf.set_index("date")
    rdf.index = pd.to_datetime(rdf.index)
    dates = rdf.index  # 改用索引日期

    rdf["ret"] = pd.to_numeric(rdf["ret"], errors="coerce").fillna(0.0)
    rdf["equity"] = pd.to_numeric(rdf["equity"], errors="coerce")

    # 基準
    bench_eq, bench_ret = load_benchmark_series(Path(args.prices), args.benchmark_symbol, dates)

    # 全期指標（原始）
    m_s = ann_metrics(rdf["ret"]); m_b = ann_metrics(bench_ret)
    m_s["max_drawdown"] = max_dd(rdf["equity"]); m_b["max_drawdown"] = max_dd(bench_eq)
    print("Full-period metrics (Strategy vs Benchmark):")
    print(f"Strategy: Sharpe={m_s['sharpe']:.2f} AnnRet={m_s['ann_return']:.2%} AnnVol={m_s['ann_vol']:.2%} MDD={m_s['max_drawdown']:.2%}")
    print(f"Benchmark({args.benchmark_symbol}): Sharpe={m_b['sharpe']:.2f} AnnRet={m_b['ann_return']:.2%} AnnVol={m_b['ann_vol']:.2%} MDD={m_b['max_drawdown']:.2%}")

    # 年度表（收益、Sharpe、MDD；基準）
    years = ydf["year"].tolist()
    bdf = bench_yearly(Path(args.prices), args.benchmark_symbol, years)
    df_year = ydf.merge(bdf, on="year", how="left")
    df_year.to_csv(out_dir / "yearly_strategy_vs_benchmark.csv", index=False, encoding="utf-8-sig")

    # 圖：年度收益柱狀
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import seaborn as sns

        x = np.arange(len(df_year)); width=0.38
        plt.figure(figsize=(max(6, 0.7*len(df_year)+2), 4.2))
        plt.bar(x - width/2, df_year["ann_return"].values, width=width, label="Strategy", color="#1f77b4")
        plt.bar(x + width/2, df_year["bench_return"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.xticks(x, df_year["year"].tolist(), rotation=0)
        plt.ylabel("Annual Return")
        plt.title("Yearly Returns (OOS)")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "yearly_returns_vs_benchmark.png", dpi=160); plt.close()

        # 年度 Sharpe
        plt.figure(figsize=(max(6, 0.7*len(df_year)+2), 4.2))
        plt.bar(x - width/2, df_year["sharpe"].values, width=width, label="Strategy", color="#2ca02c")
        plt.bar(x + width/2, df_year["bench_sharpe"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.xticks(x, df_year["year"].tolist(), rotation=0)
        plt.ylabel("Sharpe")
        plt.title("Yearly Sharpe (OOS)")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "yearly_sharpe_vs_benchmark.png", dpi=160); plt.close()

        # 年度 MDD
        plt.figure(figsize=(max(6, 0.7*len(df_year)+2), 4.2))
        plt.bar(x - width/2, df_year["max_drawdown"].values, width=width, label="Strategy", color="#1f77b4")
        plt.bar(x + width/2, df_year["bench_mdd"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
        plt.xticks(x, df_year["year"].tolist(), rotation=0)
        plt.ylabel("Max Drawdown")
        plt.title("Yearly Max Drawdown (OOS)")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "yearly_mdd_vs_benchmark.png", dpi=160); plt.close()

        # 全期累積（原始）
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        ax.plot(dates, rdf["equity"].values, label="Strategy", color="#1f77b4")
        ax.plot(dates, bench_eq.values, label=f"{args.benchmark_symbol}", color="#ff7f0e", alpha=0.85)
        ax.set_ylabel("Cumulative Equity (norm.)"); ax.set_title("Cumulative Equity (OOS)")
        ax.legend(); ax.xaxis.set_major_locator(mdates.YearLocator(base=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlim(dates.min(), dates.max()); plt.tight_layout()
        plt.savefig(out_dir / "cumulative_equity_vs_benchmark.png", dpi=160); plt.close()

        # 目標波動（若有）
        if float(args.target_vol) > 0:
            tv = float(args.target_vol)
            # 允許 10（%）或 0.10（小數）；>1 視為 %
            tv_decimal = tv/100.0 if tv > 1 else tv
            realized_vol = rdf["ret"].std(ddof=1) * np.sqrt(252)
            if realized_vol and realized_vol > 0:
                lev = tv_decimal / realized_vol
                ret_scaled = rdf["ret"] * lev
                eq_scaled = (1.0 + ret_scaled.fillna(0.0)).cumprod()
                m_s_scaled = ann_metrics(ret_scaled); m_s_scaled["max_drawdown"] = max_dd(eq_scaled)
                print(f"Strategy (target-vol={tv}%): Sharpe={m_s_scaled['sharpe']:.2f} AnnRet={m_s_scaled['ann_return']:.2%} AnnVol={m_s_scaled['ann_vol']:.2%} MDD={m_s_scaled['max_drawdown']:.2%}")

                # 年度收益（scaled）
                y_scaled = to_yearly(ret_scaled)
                y_scaled = y_scaled.rename(columns={"ret":"strat_ret_scaled"})
                y_agg = y_scaled.merge(df_year[["year","bench_return"]], on="year", how="right")
                # 圖：年度收益（scaled）
                xs = np.arange(len(y_agg))
                plt.figure(figsize=(max(6, 0.7*len(y_agg)+2), 4.2))
                plt.bar(xs - width/2, y_agg["strat_ret_scaled"].values, width=width, label=f"Strategy (TV={tv}%)", color="#1f77b4")
                plt.bar(xs + width/2, y_agg["bench_return"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
                plt.axhline(0, color="gray", linewidth=0.8)
                plt.xticks(xs, y_agg["year"].tolist(), rotation=0)
                plt.ylabel("Annual Return"); plt.title(f"Yearly Returns (Target-Vol={tv}%)")
                plt.legend(); plt.tight_layout()
                plt.savefig(out_dir / f"yearly_returns_targetVol_{int(tv)}pct.png", dpi=160); plt.close()

                # 圖：累積（scaled）
                fig, ax = plt.subplots(figsize=(9.5, 4.8))
                ax.plot(dates, eq_scaled.values, label=f"Strategy (TV={tv}%)", color="#1f77b4")
                ax.plot(dates, bench_eq.values, label=f"{args.benchmark_symbol}", color="#ff7f0e", alpha=0.85)
                ax.set_ylabel("Cumulative Equity (norm.)"); ax.set_title(f"Cumulative Equity (Target-Vol={tv}%)")
                ax.legend(); ax.xaxis.set_major_locator(mdates.YearLocator(base=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.set_xlim(dates.min(), dates.max()); plt.tight_layout()
                plt.savefig(out_dir / f"cumulative_equity_targetVol_{int(tv)}pct.png", dpi=160); plt.close()

            else:
                print("[WARN] Realized vol is zero; skip target-vol scaling.")

        print(f"[INFO] Figures saved under: {out_dir.resolve()}")

    except Exception as e:
        print(f"[WARN] Plotting skipped: {e}")
        df_year.to_csv(out_dir / "yearly_strategy_vs_benchmark.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()