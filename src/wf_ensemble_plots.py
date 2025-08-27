#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wf_ensemble_plots.py
- 讀取 wf_yearly_ensemble_pick 之輸出，產生論文/報告用圖表：
  1) yearly_returns_vs_benchmark.png
  2) yearly_sharpe_vs_benchmark.png
  3) yearly_mdd_vs_benchmark.png
  4) cumulative_equity_vs_benchmark.png
  5) yearly_returns_targetVol_10pct.png（--target-vol > 0 時）
  6) cumulative_equity_targetVol_10pct.png（--target-vol > 0 時）
  7) 其它：rolling_sharpe_252d.png、drawdown_timeseries.png、daily_return_distribution.png（若環境允許）
  
- 註解：繁體中文；print/log 英文
"""

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


# ---------- 小工具 ----------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def ann_metrics(ret: pd.Series, freq: int = 252) -> dict:
    """年化指標（日頻）。"""
    r = ret.dropna()
    if len(r) == 0:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0)
    mu = r.mean() * freq
    sd = r.std(ddof=1)
    vol = sd * np.sqrt(freq) if sd > 0 else 0.0
    sharpe = mu / vol if vol > 0 else 0.0
    return dict(ann_return=float(mu), ann_vol=float(vol), sharpe=float(sharpe))

def max_dd(eq: pd.Series) -> float:
    """最大回撤（等比），回傳負值（例如 -0.12）。"""
    if len(eq) == 0:
        return np.nan
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())

def to_yearly_returns(ret: pd.Series) -> pd.DataFrame:
    """將日報酬聚合為年度複利報酬（避免 GroupBy.apply 警告）。"""
    r = ret.copy()
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index, errors="coerce")
    comp = (1.0 + r).groupby(r.index.year).prod() - 1.0
    out = comp.rename_axis("year").reset_index(name="ret")
    out["year"] = out["year"].astype(str)
    return out

def rolling_sharpe(ret: pd.Series, window: int = 252) -> pd.Series:
    """滾動 Sharpe（年化）。"""
    r = ret.copy()
    mu = r.rolling(window).mean() * 252
    sd = r.rolling(window).std(ddof=1) * np.sqrt(252)
    return mu / sd.replace(0.0, np.nan)

def load_benchmark_series(prices_path: Path, symbol: str, dates: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    """對齊基準於策略日期；回傳 (bench_eq_norm, bench_ret_daily)。"""
    px = pd.read_pickle(prices_path)
    if symbol not in px.columns:
        raise KeyError(f"Benchmark symbol not found in prices: {symbol}")
    s = px[symbol].dropna()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(dates).ffill()
    bench_eq = s / s.iloc[0]
    bench_ret = s.pct_change().fillna(0.0)
    return bench_eq, bench_ret

def bench_yearly_metrics_from_aligned(bench_ret: pd.Series, bench_eq: pd.Series) -> pd.DataFrame:
    """由已對齊的基準日報酬與等比淨值計算年度（年報酬/Sharpe/MDD）。"""
    r = bench_ret.copy()
    e = bench_eq.copy()
    years = sorted(set(r.index.year))
    rows = []
    for y in years:
        idx = (r.index.year == y)
        ry = r[idx]
        ey = e[idx]
        if len(ry) < 3:
            rows.append(dict(year=str(y), bench_return=np.nan, bench_sharpe=np.nan, bench_mdd=np.nan))
            continue
        m = ann_metrics(ry)
        mdd = max_dd(ey / (ey.iloc[0] if len(ey) else 1.0))
        rows.append(dict(year=str(y), bench_return=float((1.0 + ry).prod() - 1.0),
                         bench_sharpe=float(m["sharpe"]), bench_mdd=float(mdd)))
    return pd.DataFrame(rows)


# ---------- 主程式 ----------

def main():
    ap = argparse.ArgumentParser(description="Plots for WF yearly ENSEMBLE outputs vs benchmark (with target-vol option).")
    ap.add_argument("--ens-root", type=str, default="reports/wf_yearly_ensemble_pick", help="Root of ensemble outputs.")
    ap.add_argument("--prices", type=str, default="data/prices.pkl", help="Daily adjusted close (wide DataFrame).")
    ap.add_argument("--benchmark-symbol", type=str, default="_GSPC", help="Benchmark symbol in prices.pkl.")
    ap.add_argument("--out-dir", type=str, default="reports/wf_yearly_ensemble_figs", help="Output directory for figures.")
    ap.add_argument("--target-vol", type=float, default=10.0, help="Target annualized vol; 10 means 10%%. Set 0 to disable.")
    ap.add_argument("--extras", action="store_true", help="If set, also output rolling Sharpe, drawdown TS, daily return distribution.")
    args = ap.parse_args()

    ens_root = Path(args.ens_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # 讀取 ensemble 輸出
    yearly_csv = ens_root / "ensemble_oos_yearly.csv"
    returns_csv = ens_root / "ensemble_oos_returns.csv"
    if not yearly_csv.exists() or not returns_csv.exists():
        print(f"[ERROR] Missing ensemble outputs: {yearly_csv} or {returns_csv}")
        return

    ydf = pd.read_csv(yearly_csv, encoding="utf-8-sig")
    # 年份統一為字串
    if "year" in ydf.columns:
        ydf["year"] = ydf["year"].astype(str)

    # 讀入日報酬與等比淨值；設日期為索引
    rdf = pd.read_csv(returns_csv, parse_dates=["date"]).sort_values("date")
    rdf["ret"] = pd.to_numeric(rdf["ret"], errors="coerce").fillna(0.0)
    rdf["equity"] = pd.to_numeric(rdf["equity"], errors="coerce")
    rdf = rdf.set_index("date")
    rdf.index = pd.to_datetime(rdf.index)
    dates = rdf.index

    # 基準序列（對齊至同日期）
    bench_eq, bench_ret = load_benchmark_series(Path(args.prices), args.benchmark_symbol, dates)

    # 全期指標（原始）
    m_s = ann_metrics(rdf["ret"]); m_b = ann_metrics(bench_ret)
    m_s["max_drawdown"] = max_dd(rdf["equity"]); m_b["max_drawdown"] = max_dd(bench_eq)
    print("Full-period metrics (Strategy ENSEMBLE vs Benchmark):")
    print(f"Strategy: Sharpe={m_s['sharpe']:.2f} AnnRet={m_s['ann_return']:.2%} AnnVol={m_s['ann_vol']:.2%} MDD={m_s['max_drawdown']:.2%}")
    print(f"Benchmark({args.benchmark_symbol}): Sharpe={m_b['sharpe']:.2f} AnnRet={m_b['ann_return']:.2%} AnnVol={m_b['ann_vol']:.2%} MDD={m_b['max_drawdown']:.2%}")

    # 年度合併表（策略已有 ann_return/ann_vol/sharpe/max_drawdown；基準需用對齊日資料計）
    b_year = bench_yearly_metrics_from_aligned(bench_ret, bench_eq)
    if "year" in b_year.columns:
        b_year["year"] = b_year["year"].astype(str)
    df_year = ydf.merge(b_year, on="year", how="left")
    df_year.to_csv(out_dir / "yearly_strategy_vs_benchmark.csv", index=False, encoding="utf-8-sig")

    # 繪圖
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.dates as mdates

        # 年度收益柱狀（策略 vs 基準）
        x = np.arange(len(df_year)); width = 0.38
        plt.figure(figsize=(max(6, 0.7*len(df_year)+2), 4.2))
        plt.bar(x - width/2, df_year["ann_return"].values, width=width, label="Strategy (Ensemble)", color="#1f77b4")
        plt.bar(x + width/2, df_year["bench_return"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.xticks(x, df_year["year"].tolist(), rotation=0)
        plt.ylabel("Annual Return")
        plt.title("Yearly Returns (OOS) - ENSEMBLE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "yearly_returns_vs_benchmark.png", dpi=160)
        plt.close()

        # 年度 Sharpe 柱狀
        plt.figure(figsize=(max(6, 0.7*len(df_year)+2), 4.2))
        plt.bar(x - width/2, df_year["sharpe"].values, width=width, label="Strategy (Ensemble)", color="#2ca02c")
        plt.bar(x + width/2, df_year["bench_sharpe"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.xticks(x, df_year["year"].tolist(), rotation=0)
        plt.ylabel("Sharpe")
        plt.title("Yearly Sharpe (OOS) - ENSEMBLE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "yearly_sharpe_vs_benchmark.png", dpi=160)
        plt.close()

        # 年度 MDD 柱狀
        plt.figure(figsize=(max(6, 0.7*len(df_year)+2), 4.2))
        plt.bar(x - width/2, df_year["max_drawdown"].values, width=width, label="Strategy (Ensemble)", color="#1f77b4")
        plt.bar(x + width/2, df_year["bench_mdd"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
        plt.xticks(x, df_year["year"].tolist(), rotation=0)
        plt.ylabel("Max Drawdown")
        plt.title("Yearly Max Drawdown (OOS) - ENSEMBLE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "yearly_mdd_vs_benchmark.png", dpi=160)
        plt.close()

        # 全期累積（策略 vs 基準）
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        ax.plot(dates, rdf["equity"].values, label="Strategy (Ensemble)", color="#1f77b4")
        ax.plot(dates, bench_eq.values, label=f"{args.benchmark_symbol}", color="#ff7f0e", alpha=0.85)
        ax.set_ylabel("Cumulative Equity (norm.)")
        ax.set_title("Cumulative Equity (OOS) - ENSEMBLE")
        ax.legend()
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlim(dates.min(), dates.max())
        plt.tight_layout()
        plt.savefig(out_dir / "cumulative_equity_vs_benchmark.png", dpi=160)
        plt.close()

        # 目標波動版本（target-vol）
        tv = float(args.target_vol)
        if tv > 0:
            tv_dec = tv/100.0 if tv > 1 else tv
            realized_vol = rdf["ret"].std(ddof=1) * np.sqrt(252)
            if realized_vol and realized_vol > 0:
                lev = tv_dec / realized_vol
                ret_scaled = rdf["ret"] * lev
                eq_scaled = (1.0 + ret_scaled.fillna(0.0)).cumprod()
                m_s_scaled = ann_metrics(ret_scaled); m_s_scaled["max_drawdown"] = max_dd(eq_scaled)
                print(f"Strategy (target-vol={tv}%): Sharpe={m_s_scaled['sharpe']:.2f} AnnRet={m_s_scaled['ann_return']:.2%} AnnVol={m_s_scaled['ann_vol']:.2%} MDD={m_s_scaled['max_drawdown']:.2%}")

                # 年度收益（scaled）
                y_scaled = to_yearly_returns(ret_scaled)
                y_scaled = y_scaled.rename(columns={"ret":"strat_ret_scaled"})
                # 以策略年度列表為主（避免 dtype 不一致）
                y_agg = y_scaled.merge(df_year[["year","bench_return"]], on="year", how="right")

                xs = np.arange(len(y_agg)); width = 0.38
                plt.figure(figsize=(max(6, 0.7*len(y_agg)+2), 4.2))
                plt.bar(xs - width/2, y_agg["strat_ret_scaled"].values, width=width, label=f"Strategy (TV={tv}%)", color="#1f77b4")
                plt.bar(xs + width/2, y_agg["bench_return"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
                plt.axhline(0, color="gray", linewidth=0.8)
                plt.xticks(xs, y_agg["year"].tolist(), rotation=0)
                plt.ylabel("Annual Return")
                plt.title(f"Yearly Returns (Target-Vol={tv}%) - ENSEMBLE")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / f"yearly_returns_targetVol_{int(tv)}pct.png", dpi=160)
                plt.close()

                # 累積（scaled）
                fig, ax = plt.subplots(figsize=(9.5, 4.8))
                ax.plot(dates, eq_scaled.values, label=f"Strategy (TV={tv}%)", color="#1f77b4")
                ax.plot(dates, bench_eq.values, label=f"{args.benchmark_symbol}", color="#ff7f0e", alpha=0.85)
                ax.set_ylabel("Cumulative Equity (norm.)")
                ax.set_title(f"Cumulative Equity (Target-Vol={tv}%) - ENSEMBLE")
                ax.legend()
                ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.set_xlim(dates.min(), dates.max())
                plt.tight_layout()
                plt.savefig(out_dir / f"cumulative_equity_targetVol_{int(tv)}pct.png", dpi=160)
                plt.close()
            else:
                print("[WARN] Realized vol is zero; skip target-vol scaling.")

        # 其它（可選）
        if args.extras:
            # 滾動 Sharpe（252d）
            rs_s = rolling_sharpe(rdf["ret"], window=252)
            rs_b = rolling_sharpe(bench_ret, window=252)
            fig, ax = plt.subplots(figsize=(9.5, 4.2))
            ax.plot(dates, rs_s, label="Strategy", color="#1f77b4")
            ax.plot(dates, rs_b, label=f"{args.benchmark_symbol}", color="#ff7f0e", alpha=0.85)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_ylabel("Rolling Sharpe (252d)")
            ax.set_title("Rolling Sharpe (OOS) - ENSEMBLE")
            ax.legend()
            ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.set_xlim(dates.min(), dates.max())
            plt.tight_layout()
            plt.savefig(out_dir / "rolling_sharpe_252d.png", dpi=160)
            plt.close()

            # 回撤時間序列
            eq_s = rdf["equity"]; peak_s = eq_s.cummax(); dd_s = eq_s/peak_s - 1.0
            peak_b = bench_eq.cummax(); dd_b = bench_eq/peak_b - 1.0
            fig, ax = plt.subplots(figsize=(9.5, 4.2))
            ax.plot(dates, dd_s.values, label="Strategy", color="#1f77b4")
            ax.plot(dates, dd_b.values, label=f"{args.benchmark_symbol}", color="#ff7f0e", alpha=0.85)
            ax.set_ylabel("Drawdown")
            ax.set_title("Drawdown (OOS) - ENSEMBLE")
            ax.legend()
            ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.set_xlim(dates.min(), dates.max())
            plt.tight_layout()
            plt.savefig(out_dir / "drawdown_timeseries.png", dpi=160)
            plt.close()

            # 日報酬分布（若 seaborn 可用）
            try:
                sns.histplot(rdf["ret"].dropna(), bins=80, stat="density", color="#1f77b4", alpha=0.4, label="Strategy")
                sns.histplot(bench_ret.dropna(), bins=80, stat="density", color="#ff7f0e", alpha=0.4, label=f"{args.benchmark_symbol}")
                sns.kdeplot(rdf["ret"].dropna(), color="#1f77b4")
                sns.kdeplot(bench_ret.dropna(), color="#ff7f0e")
                import matplotlib.pyplot as plt  # 重新引用以保存
                plt.xlabel("Daily Return")
                plt.title("Distribution of Daily Returns (Strategy vs Benchmark) - ENSEMBLE")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "daily_return_distribution.png", dpi=160)
                plt.close()
            except Exception as e:
                print(f"[WARN] Skip return distribution plot: {e}")

        print(f"[INFO] Figures saved under: {out_dir.resolve()}")

    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")


if __name__ == "__main__":
    main()