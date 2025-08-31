#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ls_compact_and_hac.py  (with Sharpe version)
- 產出 Long/Short 年化報酬的精簡 κ 表（Daily vs Weekly）
- 以每期（週或月）等權平均的 Long−Short 差做 Newey–West（HAC）檢定（各 κ 與 overall）
- 新增：以 pooled 的 Long 與 Short 序列做 JK‑M（Memmel 修正）Sharpe 檢定（各 κ 與 overall）
- 輸入：
  --daily-summary:  daily_long_short_decompose 的 long_short_summary.csv
  --weekly-summary: weekly_long_short_decompose 的 long_short_summary.csv
  --daily-ts-dir:   daily per-set CSV（long_short__L***_W***.csv）所在資料夾
  --weekly-ts-dir:  weekly per-set CSV（long_short__L***_Z***.csv）所在資料夾
  --mapping:        日↔週家族映射（"21:4,42:8,63:13,126:26,252:52"）
  --freq-out:       檢定聚合頻率（weekly 或 monthly；預設 weekly）
  --out-dir:        匯出目錄
- 輸出：
  - ls_compact_by_family.csv / .md（Δreturn 精簡表）
  - hac_tests_by_family_{daily,weekly}.csv、hac_tests_overall.txt（Δreturn）
  - jkm_tests_by_family_{daily,weekly}.csv、jkm_tests_overall.txt（ΔSharpe）
  - ls_table_with_sharpe.md / .tex（單表整合：Δreturn + ΔSharpe）
註解：繁體中文；print/log：英文
"""
import os
import re
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ===== 基礎 =====

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

def parse_mapping(s: str) -> Dict[int, int]:
    out = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        d, w = tok.split(":")
        out[int(float(d))] = int(float(w))
    return out

def k_from_freq(freq_out: str) -> int:
    return 12 if str(freq_out).lower().startswith("m") else 52

def to_dt_index(df: pd.DataFrame, date_col="date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()

def aggregate_to(r: pd.Series, freq_out: str = "weekly") -> pd.Series:
    """把日回報匯總到週或月：以連乘 − 1。"""
    r = pd.Series(r).dropna()
    if len(r) == 0: 
        return r
    if str(freq_out).lower().startswith("m"):
        return r.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0).dropna()
    else:
        return r.resample("W-FRI").apply(lambda x: (1.0 + x).prod() - 1.0).dropna()


# ===== Newey–West（HAC）t 檢定（Δreturn）=====

def _autocov(x: np.ndarray, lag: int) -> float:
    n = len(x)
    if lag >= n:
        return 0.0
    x0 = x - x.mean()
    return float(np.dot(x0[lag:], x0[:n - lag]) / n)

def nw_hac_ttest(diff: pd.Series, lag: Optional[int] = None) -> Tuple[float, float, int, float]:
    """NW（HAC）對均值的 t 檢定；回傳 t, p(two-sided), T, se。"""
    x = pd.Series(diff).dropna().values
    T = int(len(x))
    if T < 4:
        return 0.0, 1.0, T, np.nan
    if lag is None or str(lag).lower() == "auto":
        lag = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
        lag = max(1, lag)

    S = _autocov(x, 0)
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        S += 2.0 * w * _autocov(x, L)
    var_mean = S / T
    se = float(np.sqrt(max(var_mean, 1e-12)))
    mu = float(np.mean(x))
    t = float(mu / se) if se > 0 else 0.0

    from math import erf, sqrt
    cdf = 0.5 * (1.0 + erf(abs(t) / sqrt(2.0)))
    p = float(2.0 * (1.0 - cdf))
    return t, p, T, se


# ===== Jobson–Korkie（Memmel 修正）Sharpe 檢定（ΔSharpe）=====

def jkm_sharpe_test(r1: pd.Series, r2: pd.Series, k: int) -> Tuple[float, float, float]:
    """
    JK‑M 檢定（近似）：回傳 (z, p(two-sided), delta_sr_ann)。
    - r1, r2：同頻回報（未年化）
    - k：年化因子（12 或 52）
    """
    x = pd.Series(r1).dropna()
    y = pd.Series(r2).dropna()
    idx = x.index.intersection(y.index)
    x = x.reindex(idx).dropna()
    y = y.reindex(idx).dropna()
    T = int(min(len(x), len(y)))
    if T < 6:
        return 0.0, 1.0, 0.0

    mx, sx = float(x.mean()), float(x.std(ddof=1))
    my, sy = float(y.mean()), float(y.std(ddof=1))
    if sx <= 0 or sy <= 0:
        return 0.0, 1.0, 0.0

    sr1 = mx / sx
    sr2 = my / sy
    rho = float(np.corrcoef(x, y)[0, 1]) if T > 2 else 0.0

    var_jk = (1.0 / T) * (2.0 * (1.0 - rho) + 0.5 * (sr1 ** 2 + sr2 ** 2) - rho * sr1 * sr2)
    var_jkm = max(1e-12, var_jk * (1.0 - 0.5 * sr1 ** 2 - 0.5 * sr2 ** 2))
    z = float((sr1 - sr2) / np.sqrt(var_jkm))  # 這裡定義 ΔSharpe = SR_long − SR_short

    delta_sr_ann = float((sr1 - sr2) * np.sqrt(k))

    from math import erf, sqrt
    cdf = 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))
    p = float(2.0 * (1.0 - cdf))
    return z, p, delta_sr_ann


# ===== 讀取 summary 與建立 κ 表（Δreturn）=====

def read_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"formation_length","z_window","ann_return_long","ann_return_short","sharpe_long","sharpe_short"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"summary missing columns: {miss}")
    return df

def family_labels_from_summary(df: pd.DataFrame, mapping: Dict[int,int], source: str) -> pd.DataFrame:
    df = df.copy()
    if source == "daily":
        df["family"] = df["z_window"].astype(int).map(lambda z: f"{int(z)}↔{mapping.get(int(z),'?')}")
    else:
        rev = {v:k for k,v in mapping.items()}
        df["family"] = df["z_window"].astype(int).map(lambda z: f"{rev.get(int(z),'?')}↔{int(z)}")
    return df

def compact_table_by_family(daily_sum: pd.DataFrame, weekly_sum: pd.DataFrame, mapping: Dict[int,int]) -> pd.DataFrame:
    dd = family_labels_from_summary(daily_sum, mapping, "daily")
    ww = family_labels_from_summary(weekly_sum, mapping, "weekly")
    fams = sorted(set(dd["family"]).intersection(set(ww["family"])))
    dd = dd[dd["family"].isin(fams)]
    ww = ww[ww["family"].isin(fams)]

    dG = dd.groupby("family").agg(
        daily_long=("ann_return_long","mean"),
        daily_short=("ann_return_short","mean")
    ).reset_index()
    dG["daily_delta"] = dG["daily_long"] - dG["daily_short"]

    wG = ww.groupby("family").agg(
        weekly_long=("ann_return_long","mean"),
        weekly_short=("ann_return_short","mean")
    ).reset_index()
    wG["weekly_delta"] = wG["weekly_long"] - wG["weekly_short"]

    M = pd.merge(dG, wG, on="family", how="inner").sort_values("family").reset_index(drop=True)
    return M


# ===== 收集 per-set 長/短時間序列（用於 HAC/JK‑M）=====

r"""
檔名規則：
- daily:  long_short__L(\d{3})_W(\d+).csv
- weekly: long_short__L(\d{3})_Z(\d+).csv
"""

def collect_both_series(ts_dir: str, mapping: Dict[int,int], source: str) -> Tuple[Dict[str, List[pd.Series]], Dict[str, List[pd.Series]]]:
    """
    回傳 (long_map, short_map)，兩者皆為 {family: [Series(weekly or monthly aggregated), ...]}
    """
    long_map: Dict[str, List[pd.Series]] = {}
    short_map: Dict[str, List[pd.Series]] = {}

    rev = {v:k for k,v in mapping.items()}
    patt = re.compile(r"long_short__L(\d{3})_(W|Z)(\d+)\.csv", re.I)

    for fn in os.listdir(ts_dir):
        m = patt.match(fn)
        if not m:
            continue
        Lc, typ, Zw = m.group(1), m.group(2).upper(), int(m.group(3))
        if source == "daily" and typ != "W":
            continue
        if source == "weekly" and typ != "Z":
            continue
        fam = f"{Zw}↔{mapping.get(Zw,'?')}" if source == "daily" else f"{rev.get(Zw,'?')}↔{Zw}"

        fp = os.path.join(ts_dir, fn)
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        need = {"date","ret_long","ret_short"}
        if not need.issubset(set(df.columns)):
            continue
        df2 = to_dt_index(df, "date")
        long_w  = aggregate_to(df2["ret_long"],  freq_out="weekly")
        short_w = aggregate_to(df2["ret_short"], freq_out="weekly")

        long_map.setdefault(fam, []).append(long_w)
        short_map.setdefault(fam, []).append(short_w)

    return long_map, short_map


# ===== Pooled 檢定：HAC（Δreturn）與 JK‑M（ΔSharpe）=====

def pooled_hac_by_family(diff_map: Dict[str, List[pd.Series]], freq_out: str="weekly", lag="auto") -> pd.DataFrame:
    rows = []
    k = k_from_freq(freq_out)
    for fam, series_list in sorted(diff_map.items()):
        if not series_list:
            continue
        panel = pd.concat(series_list, axis=1)
        pooled = panel.mean(axis=1, skipna=True).dropna()
        if len(pooled) < 6:
            rows.append(dict(family=fam, T=len(pooled), mean_ann=np.nan, t=np.nan, p=np.nan))
            continue
        t, p, T, se = nw_hac_ttest(pooled, lag=lag)
        mean_ann = float(pooled.mean()) * k
        rows.append(dict(family=fam, T=int(T), mean_ann=float(mean_ann), t=float(t), p=float(p)))
    return pd.DataFrame(rows).sort_values("family").reset_index(drop=True)

def pooled_hac_overall(diff_map: Dict[str, List[pd.Series]], freq_out: str="weekly", lag="auto") -> Tuple[int,float,float,float]:
    all_list = []
    for fam, lst in diff_map.items():
        all_list.extend(lst)
    if not all_list:
        return 0, np.nan, np.nan, np.nan
    panel = pd.concat(all_list, axis=1)
    pooled = panel.mean(axis=1, skipna=True).dropna()
    if len(pooled) < 6:
        return len(pooled), np.nan, np.nan, np.nan
    k = k_from_freq(freq_out)
    t, p, T, se = nw_hac_ttest(pooled, lag=lag)
    mean_ann = float(pooled.mean()) * k
    return int(T), float(mean_ann), float(t), float(p)

def pooled_jkm_by_family(long_map: Dict[str, List[pd.Series]], short_map: Dict[str, List[pd.Series]], freq_out: str="weekly") -> pd.DataFrame:
    rows = []
    k = k_from_freq(freq_out)
    fams = sorted(set(long_map.keys()).union(set(short_map.keys())))
    for fam in fams:
        Ls = long_map.get(fam, [])
        Ss = short_map.get(fam, [])
        if not Ls or not Ss:
            rows.append(dict(family=fam, T=0, delta_sr_ann=np.nan, z=np.nan, p=np.nan))
            continue
        panel_L = pd.concat(Ls, axis=1)
        panel_S = pd.concat(Ss, axis=1)
        pooled_L = panel_L.mean(axis=1, skipna=True).dropna()
        pooled_S = panel_S.mean(axis=1, skipna=True).dropna()
        idx = pooled_L.index.intersection(pooled_S.index)
        pooled_L = pooled_L.reindex(idx).dropna()
        pooled_S = pooled_S.reindex(idx).dropna()
        if len(pooled_L) < 6:
            rows.append(dict(family=fam, T=len(pooled_L), delta_sr_ann=np.nan, z=np.nan, p=np.nan))
            continue
        z, p, delta_sr_ann = jkm_sharpe_test(pooled_L, pooled_S, k=k)
        rows.append(dict(family=fam, T=len(pooled_L), delta_sr_ann=float(delta_sr_ann), z=float(z), p=float(p)))
    return pd.DataFrame(rows).sort_values("family").reset_index(drop=True)

def pooled_jkm_overall(long_map: Dict[str, List[pd.Series]], short_map: Dict[str, List[pd.Series]], freq_out: str="weekly") -> Tuple[int,float,float,float]:
    all_L, all_S = [], []
    for fam in sorted(set(long_map.keys()).union(set(short_map.keys()))):
        all_L.extend(long_map.get(fam, []))
        all_S.extend(short_map.get(fam, []))
    if not all_L or not all_S:
        return 0, np.nan, np.nan, np.nan
    panel_L = pd.concat(all_L, axis=1)
    panel_S = pd.concat(all_S, axis=1)
    pooled_L = panel_L.mean(axis=1, skipna=True).dropna()
    pooled_S = panel_S.mean(axis=1, skipna=True).dropna()
    idx = pooled_L.index.intersection(pooled_S.index)
    pooled_L = pooled_L.reindex(idx).dropna()
    pooled_S = pooled_S.reindex(idx).dropna()
    if len(pooled_L) < 6:
        return len(pooled_L), np.nan, np.nan, np.nan
    k = k_from_freq("weekly")
    z, p, delta_sr_ann = jkm_sharpe_test(pooled_L, pooled_S, k=k)
    return int(len(pooled_L)), float(delta_sr_ann), float(z), float(p)


# ===== 表格輸出（Markdown / LaTeX）=====

def write_md_table_returns(df: pd.DataFrame, fp: str):
    def pct(x):
        try: return f"{float(x)*100:.2f}%"
        except: return "—"
    with open(fp, "w", encoding="utf-8") as f:
        f.write("| family | Daily‑Long | Daily‑Short | Δ(D) | Weekly‑Long | Weekly‑Short | Δ(W) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _, r in df.iterrows():
            f.write("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                r["family"], pct(r["daily_long"]), pct(r["daily_short"]), pct(r["daily_delta"]),
                pct(r["weekly_long"]), pct(r["weekly_short"]), pct(r["weekly_delta"])
            ))

def write_md_table_with_sharpe(df_ret: pd.DataFrame,
                               hac_weekly: pd.DataFrame,
                               jkm_daily: pd.DataFrame,
                               jkm_weekly: pd.DataFrame,
                               out_fp: str):
    """單表整合：Δreturn + ΔSharpe + p 值。只在 Weekly 顯示 p 值（Daily 亦可顯示，若需要可再加）。"""
    # 將 p 值合併進來
    hac_map = {r["family"]: r["p"] for _, r in hac_weekly.iterrows()}
    jkm_d_map = {r["family"]: r["p"] for _, r in jkm_daily.iterrows()}
    jkm_w_map = {r["family"]: r["p"] for _, r in jkm_weekly.iterrows()}
    dSR_map   = {r["family"]: r["delta_sr_ann"] for _, r in jkm_daily.iterrows()}
    wSR_map   = {r["family"]: r["delta_sr_ann"] for _, r in jkm_weekly.iterrows()}

    def pct(x): 
        try: return f"{float(x)*100:.2f}%"
        except: return "—"
    def ptxt(p):
        if p!=p: return "—"
        return f"{float(p):.4f}"

    with open(out_fp, "w", encoding="utf-8") as f:
        f.write("| κ family | Δ(D) | Δ(W) | HAC p(W) | ΔSharpe(D) | ΔSharpe(W) | JK‑M p(W) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _, r in df_ret.iterrows():
            fam = r["family"]
            f.write("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                fam,
                pct(r["daily_delta"]),
                pct(r["weekly_delta"]),
                ptxt(hac_map.get(fam, np.nan)),
                f"{dSR_map.get(fam, np.nan):.3f}" if fam in dSR_map and dSR_map[fam]==dSR_map[fam] else "—",
                f"{wSR_map.get(fam, np.nan):.3f}" if fam in wSR_map and wSR_map[fam]==wSR_map[fam] else "—",
                ptxt(jkm_w_map.get(fam, np.nan))
            ))

def write_tex_table_with_sharpe(df_ret: pd.DataFrame,
                                hac_weekly: pd.DataFrame,
                                jkm_daily: pd.DataFrame,
                                jkm_weekly: pd.DataFrame,
                                out_fp: str):
    hac_map = {r["family"]: r["p"] for _, r in hac_weekly.iterrows()}
    jkm_d_map = {r["family"]: r["p"] for _, r in jkm_daily.iterrows()}
    jkm_w_map = {r["family"]: r["p"] for _, r in jkm_weekly.iterrows()}
    dSR_map   = {r["family"]: r["delta_sr_ann"] for _, r in jkm_daily.iterrows()}
    wSR_map   = {r["family"]: r["delta_sr_ann"] for _, r in jkm_weekly.iterrows()}

    def pct(x):
        try: return f"{float(x)*100:.2f}\\%"
        except: return "—"

    with open(out_fp, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Long vs. Short: annualized return and Sharpe by family with tests.}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Family & $\\Delta(D)$ & $\\Delta(W)$ & HAC $p$(W) & $\\Delta SR(D)$ & $\\Delta SR(W)$ & JK--M $p$(W)\\\\\n\\midrule\n")
        for _, r in df_ret.iterrows():
            fam = r["family"]
            f.write("{} & {} & {} & {:.4f} & {} & {} & {:.4f} \\\\\n".format(
                fam,
                pct(r["daily_delta"]),
                pct(r["weekly_delta"]),
                float(hac_map.get(fam, np.nan)) if hac_map.get(fam, np.nan)==hac_map.get(fam, np.nan) else float("nan"),
                f"{dSR_map.get(fam, np.nan):.3f}" if fam in dSR_map and dSR_map[fam]==dSR_map[fam] else "—",
                f"{wSR_map.get(fam, np.nan):.3f}" if fam in wSR_map and wSR_map[fam]==wSR_map[fam] else "—",
                float(jkm_w_map.get(fam, np.nan)) if jkm_w_map.get(fam, np.nan)==jkm_w_map.get(fam, np.nan) else float("nan")
            ))
        f.write("\\bottomrule\n\\end{tabular}\n\\label{tab:ls_with_sharpe}\n\\end{table}\n")


# ===== 主流程 =====

def main():
    ap = argparse.ArgumentParser(description="Build compact κ table and run HAC/JK-M tests for Long−Short.")
    ap.add_argument("--daily-summary", type=str, required=True, help="Daily long_short_summary.csv")
    ap.add_argument("--weekly-summary", type=str, required=True, help="Weekly long_short_summary.csv")
    ap.add_argument("--daily-ts-dir", type=str, required=True, help="Folder of daily per-set long_short__*.csv")
    ap.add_argument("--weekly-ts-dir", type=str, required=True, help="Folder of weekly per-set long_short__*.csv")
    ap.add_argument("--mapping", type=str, default="21:4,42:8,63:13,126:26,252:52", help="Daily→Weekly mapping")
    ap.add_argument("--freq-out", type=str, default="weekly", choices=["weekly","monthly"], help="Aggregation frequency for tests.")
    ap.add_argument("--nw-lag", type=str, default="auto", help="NW lag (int or 'auto').")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    args = ap.parse_args()

    setup_logger("INFO")
    ensure_dir(args.out_dir)
    mapping = parse_mapping(args.mapping)

    # 讀 summary → 產 Δreturn 的 κ 表
    dsum = read_summary(args.daily_summary)
    wsum = read_summary(args.weekly_summary)
    table_ret = compact_table_by_family(dsum, wsum, mapping)
    # 輸出 Δreturn 表
    out_csv = os.path.join(args.out_dir, "ls_compact_by_family.csv")
    table_ret.to_csv(out_csv, index=False)
    out_md = os.path.join(args.out_dir, "ls_compact_by_family.md")
    write_md_table_returns(table_ret, out_md)
    print(f"[WRITE] {out_csv}")
    print(f"[WRITE] {out_md}")

    # 收集 per-set 序列
    long_d, short_d = collect_both_series(args.daily_ts_dir,  mapping, "daily")
    long_w, short_w = collect_both_series(args.weekly_ts_dir, mapping, "weekly")

    # Δreturn 的 diff_map（由 long-short 得到）
    diff_d = {fam: [L.sub(S, fill_value=0.0) for L, S in zip(long_d.get(fam, []), short_d.get(fam, []))]
              for fam in set(list(long_d.keys())+list(short_d.keys()))}
    diff_w = {fam: [L.sub(S, fill_value=0.0) for L, S in zip(long_w.get(fam, []), short_w.get(fam, []))]
              for fam in set(list(long_w.keys())+list(short_w.keys()))}

    # HAC by family
    hac_d = pooled_hac_by_family(diff_d, freq_out=args.freq_out, lag=args.nw_lag)
    hac_w = pooled_hac_by_family(diff_w, freq_out=args.freq_out, lag=args.nw_lag)
    hac_d.to_csv(os.path.join(args.out_dir, "hac_tests_by_family_daily.csv"), index=False)
    hac_w.to_csv(os.path.join(args.out_dir, "hac_tests_by_family_weekly.csv"), index=False)
    print(f"[WRITE] {os.path.join(args.out_dir, 'hac_tests_by_family_daily.csv')}")
    print(f"[WRITE] {os.path.join(args.out_dir, 'hac_tests_by_family_weekly.csv')}")

    # HAC overall
    T_d, mean_d, t_d, p_d = pooled_hac_overall(diff_d, freq_out=args.freq_out, lag=args.nw_lag)
    T_w, mean_w, t_w, p_w = pooled_hac_overall(diff_w, freq_out=args.freq_out, lag=args.nw_lag)
    with open(os.path.join(args.out_dir, "hac_tests_overall.txt"), "w", encoding="utf-8") as f:
        f.write("== Pooled HAC on (Long−Short) ==\n")
        f.write(f"Daily:  T={T_d}  mean_ann={mean_d:.4%}  t={t_d:.3f}  p(two‑sided)={p_d:.4f}\n")
        f.write(f"Weekly: T={T_w}  mean_ann={mean_w:.4%}  t={t_w:.3f}  p(two‑sided)={p_w:.4f}\n")
    print(f"[WRITE] {os.path.join(args.out_dir, 'hac_tests_overall.txt')}")

    # JK-M by family（Sharpe）
    # 上面行的別名參數為方便閱讀（python 不支援位置參數別名，這裡只是靜態提示）
    jkm_d = pooled_jkm_by_family(long_d, short_d, freq_out=args.freq_out)
    jkm_w = pooled_jkm_by_family(long_w, short_w, freq_out=args.freq_out)

    jkm_d.to_csv(os.path.join(args.out_dir, "jkm_tests_by_family_daily.csv"), index=False)
    jkm_w.to_csv(os.path.join(args.out_dir, "jkm_tests_by_family_weekly.csv"), index=False)
    print(f"[WRITE] {os.path.join(args.out_dir, 'jkm_tests_by_family_daily.csv')}")
    print(f"[WRITE] {os.path.join(args.out_dir, 'jkm_tests_by_family_weekly.csv')}")

    # JK-M overall
    Td_sr, D_delta_sr, Dz, Dp = pooled_jkm_overall(long_d, short_d, freq_out=args.freq_out)
    Tw_sr, W_delta_sr, Wz, Wp = pooled_jkm_overall(long_w, short_w, freq_out=args.freq_out)
    with open(os.path.join(args.out_dir, "jkm_tests_overall.txt"), "w", encoding="utf-8") as f:
        f.write("== Pooled JK-M on Sharpe(Long) − Sharpe(Short) ==\n")
        f.write(f"Daily:  T={Td_sr}  delta_SR_ann={D_delta_sr:.3f}  z={Dz:.3f}  p(two‑sided)={Dp:.4f}\n")
        f.write(f"Weekly: T={Tw_sr}  delta_SR_ann={W_delta_sr:.3f}  z={Wz:.3f}  p(two‑sided)={Wp:.4f}\n")
    print(f"[WRITE] {os.path.join(args.out_dir, 'jkm_tests_overall.txt')}")

    # 單表整合（Δreturn + ΔSharpe）
    write_md_table_with_sharpe(table_ret, hac_w, jkm_d, jkm_w, os.path.join(args.out_dir, "ls_table_with_sharpe.md"))
    write_tex_table_with_sharpe(table_ret, hac_w, jkm_d, jkm_w, os.path.join(args.out_dir, "ls_table_with_sharpe.tex"))
    print(f"[WRITE] {os.path.join(args.out_dir, 'ls_table_with_sharpe.md')}")
    print(f"[WRITE] {os.path.join(args.out_dir, 'ls_table_with_sharpe.tex')}")
    print("Done.")

if __name__ == "__main__":
    main()