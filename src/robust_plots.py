#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robust_plots.py (updated)
- 從 timeseries_tests 的輸出目錄（tests_results.csv、pooled_diff_series*.csv、near_hits_*.csv、pooled_bootstrap.json）
  產出論文用圖：
  1) pooled_diff_timeseries.png：橫斷面 pooled Δreturn@TV（等權與 invvar 若同時存在則雙線）
     - 若有 pooled_bootstrap.json，於圖上加註 mean/CI/p
  2) forest_effects_bootstrap.png：逐配對 Δreturn@TV 與 95% CI 的森林圖
     - 優先用 bootstrap CI；否則用 NW se 近似
  3) heatmap_winrate_by_L.png、heatmap_winrate_by_family.png：勝率（Δ>0）與平均 Δ
  4) heatmap_mean_delta_by_L_family.png：L × family 的平均 Δ 熱力圖
  5) distribution_plots.png：Δreturn@TV（%）與 ΔSharpe 分佈
  6) near_hits_bar_{nw|jkm|bs_ret|bs_sh}.png（若 near_hits_*.csv 存在）
- 註解：繁體中文；print/log：英文
"""

import os
import sys
import json
import argparse
import logging
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ===== 基本工具 =====

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

def setup_style(dpi: int = 150):
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9

def fmt_pct(x: float, d: int = 2, dash: str = "—") -> str:
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return dash
        return f"{float(x)*100:.{d}f}%"
    except Exception:
        return dash


# ===== 輔助讀取 =====

def read_tests(results_dir: str) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series], dict]:
    """
    讀 tests_results.csv 與 pooled_diff_series*.csv；回傳：
    - res_df（欄名轉小寫）
    - pooled_equal（等權 pooled Δ；未年化）
    - pooled_invvar（invvar pooled Δ；未年化）
    - meta：freq_out、target_vol、pooled_bootstrap（若有）
    """
    fp = os.path.join(results_dir, "tests_results.csv")
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"tests_results.csv not found in {results_dir}")
    res = pd.read_csv(fp)
    res.columns = [c.strip().lower() for c in res.columns]

    # 讀 pooled 等權
    pooled_equal, pooled_invvar = None, None
    for name in ["pooled_diff_series_equal.csv", "pooled_diff_series.csv"]:
        p = os.path.join(results_dir, name)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            if "date" in cols and "pooled_diff" in cols:
                s = df[[cols["date"], cols["pooled_diff"]]].copy()
                s[cols["date"]] = pd.to_datetime(s[cols["date"]])
                s = s.set_index(cols["date"]).sort_index()[cols["pooled_diff"]]
                pooled_equal = s
                break
    # 讀 pooled invvar
    p = os.path.join(results_dir, "pooled_diff_series_invvar.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        if "date" in cols and "pooled_diff" in cols:
            s = df[[cols["date"], cols["pooled_diff"]]].copy()
            s[cols["date"]] = pd.to_datetime(s[cols["date"]])
            s = s.set_index(cols["date"]).sort_index()[cols["pooled_diff"]]
            pooled_invvar = s

    # 讀 meta
    meta = {}
    if "freq_out" in res.columns:
        meta["freq_out"] = str(res["freq_out"].iloc[0])
    if "target_vol" in res.columns:
        meta["target_vol"] = float(res["target_vol"].iloc[0])

    fp_bs = os.path.join(results_dir, "pooled_bootstrap.json")
    if os.path.isfile(fp_bs):
        try:
            meta["pooled_bootstrap"] = json.load(open(fp_bs, "r", encoding="utf-8"))
        except Exception:
            pass

    return res, pooled_equal, pooled_invvar, meta


def k_from_freq(freq_out: str) -> int:
    return 12 if str(freq_out).lower().startswith("m") else 52


# ===== 繪圖 =====

def plot_pooled_timeseries(out_dir: str,
                           pooled_equal: Optional[pd.Series],
                           pooled_invvar: Optional[pd.Series],
                           freq_out: str,
                           pooled_bs: Optional[dict] = None,
                           roll_win: Optional[int] = None,
                           title_prefix: str = "Pooled Δreturn@TV"):
    """畫 pooled Δ 序列；若有 bootstrap 統計（annualized CI/p），於圖標題加註。"""
    if (pooled_equal is None) and (pooled_invvar is None):
        logging.info("No pooled series found; skip pooled plot.")
        return

    if roll_win is None:
        roll_win = 26 if str(freq_out).lower().startswith("w") else 12

    plt.figure(figsize=(10, 4.6))

    title_suffix = []
    if pooled_bs and isinstance(pooled_bs, dict):
        eq = pooled_bs.get("equal", {})
        iv = pooled_bs.get("invvar", {})
        if "mean_ann" in eq:
            title_suffix.append(f"EQ: mean={fmt_pct(eq.get('mean_ann', np.nan))} CI[{fmt_pct(eq.get('ci_lo', np.nan))},{fmt_pct(eq.get('ci_hi', np.nan))}] p={eq.get('p', np.nan):.3f}")
        if "mean_ann" in iv:
            title_suffix.append(f"IV: mean={fmt_pct(iv.get('mean_ann', np.nan))} CI[{fmt_pct(iv.get('ci_lo', np.nan))},{fmt_pct(iv.get('ci_hi', np.nan))}] p={iv.get('p', np.nan):.3f}")

    if pooled_equal is not None:
        s = pooled_equal
        plt.plot(s.index, s.values*100.0, label=f"Equal-weight (per-period, mean={fmt_pct(s.mean())})", alpha=0.85)
        rm = s.rolling(roll_win, min_periods=max(2, roll_win//3)).mean()
        plt.plot(rm.index, rm.values*100.0, label=f"Equal-weight rolling mean ({roll_win})", linewidth=1.8)
    if pooled_invvar is not None:
        s2 = pooled_invvar
        plt.plot(s2.index, s2.values*100.0, label=f"InvVar-weight (per-period, mean={fmt_pct(s2.mean())})", alpha=0.85)
        rm2 = s2.rolling(roll_win, min_periods=max(2, roll_win//3)).mean()
        plt.plot(rm2.index, rm2.values*100.0, label=f"InvVar-weight rolling mean ({roll_win})", linewidth=1.8)

    plt.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ttl = f"{title_prefix} | {freq_out}"
    if title_suffix:
        ttl += "\n" + " | ".join(title_suffix)
    plt.title(ttl)
    plt.xlabel("Date"); plt.ylabel("Per-period Δreturn (%, not annualized)")
    plt.legend()
    plt.tight_layout()
    fp = os.path.join(out_dir, "pooled_diff_timeseries.png")
    plt.savefig(fp); plt.close()
    print(f"[WRITE] {fp}")


def plot_forest_effects(out_dir: str, res: pd.DataFrame, top_n: int = 25):
    """森林圖：逐配對年化 Δ 與 95% CI（優先 bootstrap）。"""
    df = res.copy()
    # 若沒有 bootstrap CI，改用 NW se 近似
    has_bs_any = df[["bs_ret_ci_lo","bs_ret_ci_hi"]].notna().all(axis=1).any() if {"bs_ret_ci_lo","bs_ret_ci_hi"}.issubset(df.columns) else False
    if not has_bs_any and "nw_se" in df.columns and "freq_out" in df.columns:
        k = k_from_freq(df["freq_out"].iloc[0])
        df["bs_ret_ci_lo"] = df["delta_ann_return"] - 1.96 * df["nw_se"] * k
        df["bs_ret_ci_hi"] = df["delta_ann_return"] + 1.96 * df["nw_se"] * k

    # 取最大的正向效果（也可以改為按絕對值）
    df = df.sort_values("delta_ann_return", ascending=True).tail(top_n)
    if df.empty:
        logging.info("No rows for forest plot."); return

    ylab = [f"L{float(L):.1f} {fam}" for L, fam in zip(df.get("l", df.get("L", df.index)), df["family"])]
    y = np.arange(len(df))
    x = df["delta_ann_return"] * 100.0
    lo = df["bs_ret_ci_lo"] * 100.0
    hi = df["bs_ret_ci_hi"] * 100.0

    plt.figure(figsize=(8.2, max(6, 0.28*len(df)+1)))
    plt.errorbar(x, y, xerr=[x - lo, hi - x], fmt='o', color="tab:blue", ecolor="tab:blue", elinewidth=1.2, capsize=3)
    plt.axvline(0.0, color="gray", lw=0.8, alpha=0.6)
    plt.yticks(y, ylab)
    plt.xlabel("Δreturn@TV (annualized, %)")
    plt.title("Largest effects (Δreturn@TV) with 95% CI")
    plt.tight_layout()
    fp = os.path.join(out_dir, "forest_effects_bootstrap.png")
    plt.savefig(fp); plt.close()
    print(f"[WRITE] {fp}")

def plot_heatmaps(out_dir: str, res: pd.DataFrame):
    df = res.copy()

    # 自動偵測 formation length 欄位名稱（l 或 L）
    colL = "l" if "l" in df.columns else ("L" if "L" in df.columns else None)
    if colL is None:
        logging.warning("No formation length column (l/L); skip heatmaps.")
        return

    # by formation length：勝率（Δ>0）與平均 Δ
    gL = df.groupby(colL)
    wrL = gL["delta_ann_return"].apply(lambda s: float((s > 0).mean()) * 100.0).to_frame("win%")
    muL = gL["delta_ann_return"].mean().to_frame("meanΔ")
    htL = wrL.join(muL)

    plt.figure(figsize=(5.8, max(3.6, 0.5*len(htL))))
    sns.heatmap(
        htL[["win%","meanΔ"]].assign(**{"meanΔ": htL["meanΔ"]*100.0}),
        annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label":"Value"}
    )
    plt.title("Win rate (%) and mean Δ by formation length")
    plt.tight_layout()
    fp = os.path.join(out_dir, "heatmap_winrate_by_L.png")
    plt.savefig(fp); plt.close()
    print(f"[WRITE] {fp}")

    # by family：勝率與平均 Δ
    gf = df.groupby("family")
    wrF = gf["delta_ann_return"].apply(lambda s: float((s > 0).mean()) * 100.0).to_frame("win%")
    muF = gf["delta_ann_return"].mean().to_frame("meanΔ")
    htF = wrF.join(muF)

    plt.figure(figsize=(6.6, max(3.6, 0.5*len(htF))))
    sns.heatmap(
        htF[["win%","meanΔ"]].assign(**{"meanΔ": htF["meanΔ"]*100.0}),
        annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label":"Value"}
    )
    plt.title("Win rate (%) and mean Δ by family (D↔W)")
    plt.tight_layout()
    fp = os.path.join(out_dir, "heatmap_winrate_by_family.png")
    plt.savefig(fp); plt.close()
    print(f"[WRITE] {fp}")

    # L × family 的平均 Δ（年化 %）
    pv = df.pivot_table(
        index=colL, columns="family", values="delta_ann_return", aggfunc="mean"
    )
    if pv is not None and pv.shape[0] > 0:
        plt.figure(figsize=(max(7, 0.5*pv.shape[1]+2), max(4, 0.5*pv.shape[0]+1)))
        sns.heatmap(
            pv*100.0, annot=True, fmt=".2f", cmap="RdYlGn", center=0.0,
            cbar_kws={"label":"Mean Δreturn@TV (annualized, %)"}
        )
        plt.title("Mean Δ by (L × family)")
        plt.tight_layout()
        fp = os.path.join(out_dir, "heatmap_mean_delta_by_L_family.png")
        plt.savefig(fp); plt.close()
        print(f"[WRITE] {fp}")

def plot_distributions(out_dir: str, res: pd.DataFrame):
    plt.figure(figsize=(10, 4.2))
    plt.subplot(1,2,1)
    sns.histplot(res["delta_ann_return"]*100.0, bins=40, kde=True, color="tab:blue")
    plt.axvline(0.0, color="gray", lw=0.8, alpha=0.6)
    plt.title("Δreturn@TV (annualized, %) distribution")
    plt.subplot(1,2,2)
    sns.histplot(res["delta_sharpe"], bins=40, kde=True, color="tab:orange")
    plt.axvline(0.0, color="gray", lw=0.8, alpha=0.6)
    plt.title("ΔSharpe distribution")
    plt.tight_layout()
    fp = os.path.join(out_dir, "distribution_plots.png")
    plt.savefig(fp); plt.close()
    print(f"[WRITE] {fp}")


def plot_near_hits(out_dir: str, results_dir: str, kind: str = "nw", top_k: int = 10):
    """若 near_hits_{kind}.csv 存在則畫條圖（大小寫無關欄位）。"""
    fp = os.path.join(results_dir, f"near_hits_{kind}.csv")
    if not os.path.isfile(fp):
        logging.info(f"near_hits file not found: {fp}")
        return
    df = pd.read_csv(fp)
    if df.empty:
        return
    df.columns = [c.strip().lower() for c in df.columns]

    plt.figure(figsize=(8.2, max(4, 0.36*len(df))))
    y = np.arange(len(df))
    labels = [f"L{float(L):.1f} {fam}" for L, fam in zip(df.get("l", df.get("L", df.index)), df.get("family", [""]*len(df)))]

    if "delta_ann_return" in df.columns:
        x = df["delta_ann_return"]*100.0
        plt.barh(y, x, color="tab:blue", alpha=0.8)
        if {"bs_ret_ci_lo","bs_ret_ci_hi"}.issubset(df.columns):
            lo = df["bs_ret_ci_lo"]*100.0
            hi = df["bs_ret_ci_hi"]*100.0
            for i in range(len(df)):
                plt.plot([lo.iloc[i], hi.iloc[i]], [y[i], y[i]], color="k", lw=1.2)
        xlabel = "Δreturn@TV (annualized, %)"
    else:
        x = df.get("delta_sharpe", pd.Series([0]*len(df)))
        plt.barh(y, x, color="tab:orange", alpha=0.8)
        xlabel = "ΔSharpe"

    plt.yticks(y, labels)
    plt.xlabel(xlabel)
    plt.title(f"Near-hits ({kind}) Top-{min(top_k,len(df))}")
    plt.axvline(0.0, color="gray", lw=0.8, alpha=0.6)
    plt.tight_layout()
    fp2 = os.path.join(out_dir, f"near_hits_bar_{kind}.png")
    plt.savefig(fp2); plt.close()
    print(f"[WRITE] {fp2}")


# ===== 主程式 =====

def main():
    ap = argparse.ArgumentParser(description="Generate robustness plots from timeseries_tests outputs.")
    ap.add_argument("--tests-dir", type=str, required=True, help="Directory with tests_results.csv (+ optional pooled/near_hits files).")
    ap.add_argument("--topn-forest", type=int, default=25, help="Number of pairs to display in forest plot.")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = ap.parse_args()

    setup_logger("INFO")
    setup_style(args.dpi)

    results_dir = args.tests_dir
    if not os.path.isdir(results_dir):
        print(f"[ERROR] tests-dir not found: {results_dir}")
        sys.exit(2)

    try:
        res, pooled_equal, pooled_invvar, meta = read_tests(results_dir)
    except Exception as e:
        print(f"[ERROR] failed to load tests outputs: {repr(e)}")
        sys.exit(2)

    out_dir = os.path.join(results_dir, "plots")
    ensure_dir(out_dir)

    freq_out = str(meta.get("freq_out","(unknown)"))
    pooled_bs = meta.get("pooled_bootstrap", None)

    plot_pooled_timeseries(out_dir, pooled_equal, pooled_invvar, freq_out=freq_out, pooled_bs=pooled_bs)

    # Forest（若樣本少，topn 會自動截短）
    if "delta_ann_return" in res.columns:
        plot_forest_effects(out_dir, res, top_n=int(args.topn_forest))

    plot_heatmaps(out_dir, res)
    plot_distributions(out_dir, res)

    # Near-hits（若存在）
    for kind in ["nw","jkm","bs_ret","bs_sh"]:
        plot_near_hits(out_dir, results_dir, kind=kind)

    print("Done.")

if __name__ == "__main__":
    main()