#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_tests_results.py
- 合併多個 timeseries_tests 的輸出資料夾為一張總表，便於貼到論文或分享。
- 每個資料夾至少需有 tests_results.csv；若有 pooled_diff_series_*.csv 與 pooled_bootstrap.json 也會一起納入摘要。
- 輸出：
  - merged_summary.csv
  - merged_summary.md（Markdown 表格）
  - merged_summary.txt（等寬對齊文字表）
註解：繁體中文；print/log：英文
"""

import os
import sys
import json
import glob
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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

def fmt_pct(x: float, d: int = 2) -> str:
    try:
        return f"{float(x)*100:.{d}f}%"
    except Exception:
        return "NA"

def annualization_k(freq_out: str) -> int:
    return 12 if str(freq_out).lower().startswith("m") else 52

# Newey–West（HAC）t 檢定（複用 timeseries_tests 的版本）
def _autocov(x: np.ndarray, lag: int) -> float:
    n = len(x)
    if lag >= n:
        return 0.0
    x0 = x - x.mean()
    return float(np.dot(x0[lag:], x0[:n - lag]) / n)

def nw_hac_ttest(diff: pd.Series, lag: Optional[int] = None) -> Tuple[float, float, int, float]:
    x = pd.Series(diff).dropna().values
    T = int(len(x))
    if T < 4:
        return 0.0, 1.0, T, np.nan
    if lag is None or str(lag).lower() == "auto":
        lag = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
        lag = max(1, lag)
    gamma0 = _autocov(x, 0)
    S = gamma0
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        gammaL = _autocov(x, L)
        S += 2.0 * w * gammaL
    var_mean = S / T
    se = float(np.sqrt(max(var_mean, 1e-12)))
    mu_hat = float(np.mean(x))
    t_stat = float(mu_hat / se) if se > 0 else 0.0
    from math import erf, sqrt
    cdf = 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0)))
    p = float(2.0 * (1.0 - cdf))
    return t_stat, p, T, se

def aggregate_returns(r: pd.Series, freq_out: str = "monthly") -> pd.Series:
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return r
    if str(freq_out).lower().startswith("m"):
        out = r.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    else:
        out = r.resample("W-FRI").apply(lambda x: (1.0 + x).prod() - 1.0)
    return out.dropna()

def to_datetime_index(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col])
        df = df.set_index(col)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df.sort_index()


# ===== 合併主邏輯 =====

def load_results_dir(d: str) -> Optional[Dict[str, object]]:
    """
    讀單一 results 資料夾，回傳摘要欄位字典；若缺 tests_results.csv 則回 None。
    """
    res_csv = os.path.join(d, "tests_results.csv")
    if not os.path.isfile(res_csv):
        logging.warning(f"tests_results.csv not found: {d}")
        return None

    df = pd.read_csv(res_csv)
    if df.empty:
        logging.warning(f"Empty tests_results.csv: {d}")
        return None

    # 基本欄位
    scenario = os.path.basename(os.path.normpath(d))
    freq_out = str(df["freq_out"].iloc[0]) if "freq_out" in df.columns else ""
    target_vol = float(df["target_vol"].iloc[0]) if "target_vol" in df.columns else np.nan
    pairs_tested = int(len(df))

    # 平均效果
    mean_delta_ret = float(df["delta_ann_return"].mean()) if "delta_ann_return" in df.columns else np.nan
    mean_delta_sh  = float(df["delta_sharpe"].mean()) if "delta_sharpe" in df.columns else np.nan

    # 顯著比例（raw p<0.05）
    def sig_rate(col):
        return float((pd.to_numeric(df[col], errors="coerce") < 0.05).mean()) if col in df.columns else np.nan

    sig_nw     = sig_rate("nw_p")
    sig_jkm    = sig_rate("jkm_p")
    sig_bs_ret = sig_rate("bs_ret_p")
    sig_bs_sh  = sig_rate("bs_sh_p")

    # 最小 p
    def min_p(col):
        return float(pd.to_numeric(df[col], errors="coerce").min()) if col in df.columns else np.nan

    minp_nw     = min_p("nw_p")
    minp_jkm    = min_p("jkm_p")
    minp_bs_ret = min_p("bs_ret_p")
    minp_bs_sh  = min_p("bs_sh_p")

    # Pooled：讀等權 / invvar 差序列並重算 NW；若無則留空
    pooled_equal = None
    pooled_inv   = None
    for name in ["pooled_diff_series_equal.csv", "pooled_diff_series.csv"]:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            tmp = pd.read_csv(p)
            cols = {c.lower(): c for c in tmp.columns}
            if "date" in cols and "pooled_diff" in cols:
                s = tmp[[cols["date"], cols["pooled_diff"]]].copy()
                s[cols["date"]] = pd.to_datetime(s[cols["date"]])
                s = s.set_index(cols["date"]).sort_index()[cols["pooled_diff"]]
                pooled_equal = s
                break
    p = os.path.join(d, "pooled_diff_series_invvar.csv")
    if os.path.isfile(p):
        tmp = pd.read_csv(p)
        cols = {c.lower(): c for c in tmp.columns}
        if "date" in cols and "pooled_diff" in cols:
            s = tmp[[cols["date"], cols["pooled_diff"]]].copy()
            s[cols["date"]] = pd.to_datetime(s[cols["date"]])
            s = s.set_index(cols["date"]).sort_index()[cols["pooled_diff"]]
            pooled_inv = s

    k = annualization_k(freq_out)
    pooled_eq_T = pooled_eq_mean = pooled_eq_t = pooled_eq_p = pooled_eq_se = np.nan
    pooled_iv_T = pooled_iv_mean = pooled_iv_t = pooled_iv_p = pooled_iv_se = np.nan

    if pooled_equal is not None and len(pooled_equal) >= 4:
        t, pval, T, se = nw_hac_ttest(pooled_equal, lag="auto")
        pooled_eq_T, pooled_eq_t, pooled_eq_p, pooled_eq_se = T, t, pval, se
        pooled_eq_mean = float(pooled_equal.mean()) * k
    if pooled_inv is not None and len(pooled_inv) >= 4:
        t, pval, T, se = nw_hac_ttest(pooled_inv, lag="auto")
        pooled_iv_T, pooled_iv_t, pooled_iv_p, pooled_iv_se = T, t, pval, se
        pooled_iv_mean = float(pooled_inv.mean()) * k
    else:
        pooled_iv_mean = np.nan

    # 若有 pooled_bootstrap.json（含 equal/invvar CI 與 p）
    bs_eq_lo = bs_eq_hi = bs_eq_p = np.nan
    bs_iv_lo = bs_iv_hi = bs_iv_p = np.nan
    fp_bs = os.path.join(d, "pooled_bootstrap.json")
    if os.path.isfile(fp_bs):
        try:
            data = json.load(open(fp_bs, "r", encoding="utf-8"))
            if isinstance(data, dict):
                if "equal" in data:
                    bs_eq_lo = float(data["equal"].get("ci_lo", np.nan))
                    bs_eq_hi = float(data["equal"].get("ci_hi", np.nan))
                    bs_eq_p  = float(data["equal"].get("p", np.nan))
                if "invvar" in data:
                    bs_iv_lo = float(data["invvar"].get("ci_lo", np.nan))
                    bs_iv_hi = float(data["invvar"].get("ci_hi", np.nan))
                    bs_iv_p  = float(data["invvar"].get("p", np.nan))
        except Exception:
            pass

    return dict(
        scenario=scenario,
        freq_out=freq_out,
        target_vol=float(target_vol),
        pairs_tested=int(pairs_tested),
        mean_delta_ret=float(mean_delta_ret),
        mean_delta_sharpe=float(mean_delta_sh),

        sig_rate_nw=float(sig_nw),
        sig_rate_jkm=float(sig_jkm),
        sig_rate_bs_ret=float(sig_bs_ret),
        sig_rate_bs_sh=float(sig_bs_sh),

        min_p_nw=float(minp_nw),
        min_p_jkm=float(minp_jkm),
        min_p_bs_ret=float(minp_bs_ret),
        min_p_bs_sh=float(minp_bs_sh),

        pooled_eq_T=float(pooled_eq_T),
        pooled_eq_mean_ann=float(pooled_eq_mean),
        pooled_eq_t=float(pooled_eq_t),
        pooled_eq_p=float(pooled_eq_p),
        pooled_eq_se=float(pooled_eq_se),

        pooled_iv_T=float(pooled_iv_T),
        pooled_iv_mean_ann=float(pooled_iv_mean),
        pooled_iv_t=float(pooled_iv_t),
        pooled_iv_p=float(pooled_iv_p),
        pooled_iv_se=float(pooled_iv_se),

        pooled_bs_eq_lo=float(bs_eq_lo),
        pooled_bs_eq_hi=float(bs_eq_hi),
        pooled_bs_eq_p=float(bs_eq_p),
        pooled_bs_iv_lo=float(bs_iv_lo),
        pooled_bs_iv_hi=float(bs_iv_hi),
        pooled_bs_iv_p=float(bs_iv_p),

        dir=os.path.abspath(d)
    )


def write_markdown_table(df: pd.DataFrame, fp: str):
    cols = [
        "scenario","freq_out","target_vol","pairs_tested",
        "mean_delta_ret","mean_delta_sharpe",
        "sig_rate_nw","sig_rate_jkm","sig_rate_bs_ret","sig_rate_bs_sh",
        "pooled_eq_T","pooled_eq_mean_ann","pooled_eq_t","pooled_eq_p",
        "pooled_iv_T","pooled_iv_mean_ann","pooled_iv_t","pooled_iv_p"
    ]
    cols = [c for c in cols if c in df.columns]
    with open(fp, "w", encoding="utf-8") as f:
        # header
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"]*len(cols)) + "|\n")
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if c in ("target_vol","mean_delta_ret","mean_delta_sharpe",
                         "sig_rate_nw","sig_rate_jkm","sig_rate_bs_ret","sig_rate_bs_sh",
                         "pooled_eq_mean_ann","pooled_iv_mean_ann"):
                    vals.append(fmt_pct(v))
                elif c in ("pooled_eq_p","pooled_iv_p"):
                    vals.append(f"{v:.4f}" if np.isfinite(v) else "NA")
                else:
                    vals.append(f"{int(v)}" if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer()) else f"{v}")
            f.write("| " + " | ".join(vals) + " |\n")

def write_txt_table(df: pd.DataFrame, fp: str):
    cols = [
        ("scenario", 28), ("freq_out", 8), ("target_vol", 10), ("pairs_tested", 12),
        ("mean_delta_ret", 14), ("mean_delta_sharpe", 14),
        ("sig_rate_nw", 12), ("sig_rate_jkm", 12), ("sig_rate_bs_ret", 14), ("sig_rate_bs_sh", 14),
        ("pooled_eq_T", 11), ("pooled_eq_mean_ann", 16), ("pooled_eq_t", 10), ("pooled_eq_p", 10)
    ]
    def fmt(c, v):
        if c in ("target_vol","mean_delta_ret","mean_delta_sharpe","sig_rate_nw","sig_rate_jkm","sig_rate_bs_ret","sig_rate_bs_sh","pooled_eq_mean_ann"):
            return fmt_pct(v)
        if c in ("pooled_eq_p",):
            return f"{v:.4f}" if np.isfinite(v) else "NA"
        return str(int(v)) if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer()) else str(v)

    with open(fp, "w", encoding="utf-8") as f:
        # header
        header = "  ".join(n.ljust(w) for (n, w) in cols)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for _, r in df.iterrows():
            line = "  ".join(fmt(n, r[n]).ljust(w) for (n, w) in cols if n in df.columns)
            f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description="Merge multiple timeseries_tests outputs into one summary.")
    ap.add_argument("--dirs", type=str, nargs="*", default=[], help="List of tests result directories.")
    ap.add_argument("--glob", type=str, default="", help="Glob pattern to collect result directories (e.g., 'reports/robust_test/tests_*').")
    ap.add_argument("--out", type=str, default="reports/robust_test/merged_summary", help="Output folder for merged summary.")
    args = ap.parse_args()

    setup_logger("INFO")
    ensure_dir(args.out)

    # 收集資料夾
    dirs: List[str] = []
    if args.glob:
        dirs.extend(sorted([p for p in glob.glob(args.glob) if os.path.isdir(p)]))
    if args.dirs:
        dirs.extend([p for p in args.dirs if os.path.isdir(p)])

    dirs = sorted(list(dict.fromkeys(dirs)))
    if not dirs:
        print("[ERROR] No valid directories provided. Use --glob or --dirs.")
        sys.exit(2)

    rows = []
    for d in dirs:
        try:
            rec = load_results_dir(d)
            if rec:
                rows.append(rec)
        except Exception as e:
            logging.warning(f"Skip {d} due to error: {repr(e)}")

    if not rows:
        print("[ERROR] Nothing merged."); sys.exit(2)

    df = pd.DataFrame(rows).sort_values(["scenario"]).reset_index(drop=True)
    out_csv = os.path.join(args.out, "merged_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"[WRITE] {out_csv}")

    out_md = os.path.join(args.out, "merged_summary.md")
    write_markdown_table(df, out_md)
    print(f"[WRITE] {out_md}")

    out_txt = os.path.join(args.out, "merged_summary.txt")
    write_txt_table(df, out_txt)
    print(f"[WRITE] {out_txt}")

    print("Done.")

if __name__ == "__main__":
    main()