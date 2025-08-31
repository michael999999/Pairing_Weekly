#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ls_compact_and_hac.py
- 產出 Long/Short 年化報酬的精簡 κ 表（Daily vs Weekly）
- 以每期（週頻）等權平均的 Long−Short 差做 Newey–West 檢定（各 κ 與 overall）
- 輸入：
  --daily-summary:  daily_long_short_decompose 輸出的 long_short_summary.csv
  --weekly-summary: weekly_long_short_decompose 輸出的 long_short_summary.csv
  --daily-ts-dir:   daily 的 per-set CSV（long_short__L***_W***.csv）所在資料夾
  --weekly-ts-dir:  weekly 的 per-set CSV（long_short__L***_Z***.csv）所在資料夾
  --mapping:        日↔週家族映射（"21:4,42:8,63:13,126:26,252:52"）
  --freq-out:       檢定的聚合頻率（weekly 或 monthly；預設 weekly）
  --out-dir:        匯出表格與檢定結果目錄
- 註解：繁體中文；print/log：英文
"""
import os
import re
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

def setup_logger(level="INFO"):
    logger = logging.getLogger()
    for h in list(logger.handlers): logger.removeHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(message)s","%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def parse_mapping(s: str) -> Dict[int,int]:
    out = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok: continue
        d, w = tok.split(":")
        out[int(float(d))] = int(float(w))
    return out

def k_from_freq(freq_out: str) -> int:
    return 12 if freq_out.lower().startswith("m") else 52

# ===== Newey–West（HAC）t 檢定 =====
def _autocov(x: np.ndarray, lag: int) -> float:
    n = len(x)
    if lag >= n: return 0.0
    x0 = x - x.mean()
    return float(np.dot(x0[lag:], x0[:n-lag]) / n)

def nw_hac_ttest(diff: pd.Series, lag: Optional[int] = None) -> Tuple[float,float,int,float]:
    x = pd.Series(diff).dropna().values
    T = int(len(x))
    if T < 4: return 0.0, 1.0, T, np.nan
    if lag is None or str(lag).lower()=="auto":
        lag = int(np.floor(4.0 * (T/100.0)**(2.0/9.0))); lag = max(1, lag)
    S = _autocov(x, 0)
    for L in range(1, lag+1):
        w = 1.0 - L/(lag+1.0)
        S += 2.0 * w * _autocov(x, L)
    var_mean = S / T
    se = float(np.sqrt(max(var_mean, 1e-12)))
    mu = float(np.mean(x))
    t = float(mu / se) if se>0 else 0.0
    from math import erf, sqrt
    cdf = 0.5 * (1.0 + erf(abs(t)/sqrt(2.0)))
    p = float(2.0*(1.0 - cdf))
    return t, p, T, se

def to_dt_index(df: pd.DataFrame, date_col="date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()

def aggregate_to(r: pd.Series, freq_out: str="weekly") -> pd.Series:
    r = pd.Series(r).dropna()
    if len(r)==0: return r
    if freq_out.lower().startswith("m"):
        return r.resample("ME").apply(lambda x: (1.0+x).prod()-1.0).dropna()
    else:
        return r.resample("W-FRI").apply(lambda x: (1.0+x).prod()-1.0).dropna()

def read_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"formation_length","z_window","ann_return_long","ann_return_short"}
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"summary missing columns: {miss}")
    return df

def family_labels_from_summary(df: pd.DataFrame, mapping: Dict[int,int], source: str) -> pd.DataFrame:
    df = df.copy()
    if source=="daily":
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

    M = pd.merge(dG, wG, on="family", how="inner")
    # 轉百分比顯示友善（輸出 CSV 保留小數）
    return M.sort_values("family").reset_index(drop=True)

# 掃描 per-set 長短邊 CSV，組家庭 pooled 差序列
def collect_diff_series(ts_dir: str, mapping: Dict[int,int], source: str) -> Dict[str, List[pd.Series]]:
    r"""
    回傳 {family: [Series(weekly-aggregated diff), ...]}
    檔名規則：
      - daily:  long_short__L(\d{3})_W(\d+).csv
      - weekly: long_short__L(\d{3})_Z(\d+).csv
    """
    out: Dict[str, List[pd.Series]] = {}
    rev = {v:k for k,v in mapping.items()}
    patt = re.compile(r"long_short__L(\d{3})_(W|Z)(\d+)\.csv", re.I)
    for fn in os.listdir(ts_dir):
        m = patt.match(fn)
        if not m: continue
        Lc, typ, Zw = m.group(1), m.group(2).upper(), int(m.group(3))
        L = float(int(Lc)/100.0)
        if source=="daily" and typ!="W": continue
        if source=="weekly" and typ!="Z": continue
        fam = f"{Zw}↔{mapping.get(Zw,'?')}" if source=="daily" else f"{rev.get(Zw,'?')}↔{Zw}"

        fp = os.path.join(ts_dir, fn)
        df = pd.read_csv(fp)
        if "date" not in df.columns or "ret_long" not in df.columns or "ret_short" not in df.columns:
            continue
        df2 = to_dt_index(df, "date")
        diff = df2["ret_long"].sub(df2["ret_short"], fill_value=0.0)
        diff_w = aggregate_to(diff, freq_out="weekly")
        out.setdefault(fam, []).append(diff_w)
    return out

def pooled_hac_by_family(diff_map: Dict[str, List[pd.Series]], freq_out: str="weekly", lag="auto") -> pd.DataFrame:
    rows = []
    k = k_from_freq(freq_out)
    for fam, series_list in sorted(diff_map.items()):
        if not series_list: continue
        panel = pd.concat(series_list, axis=1)  # 對齊（日期聯集）
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
    if not all_list: return 0, np.nan, np.nan, np.nan
    panel = pd.concat(all_list, axis=1)
    pooled = panel.mean(axis=1, skipna=True).dropna()
    if len(pooled) < 6: return len(pooled), np.nan, np.nan, np.nan
    k = k_from_freq(freq_out)
    t, p, T, se = nw_hac_ttest(pooled, lag=lag)
    mean_ann = float(pooled.mean()) * k
    return int(T), float(mean_ann), float(t), float(p)

def write_md_table(df: pd.DataFrame, fp: str):
    cols = ["family","daily_long","daily_short","daily_delta","weekly_long","weekly_short","weekly_delta"]
    df2 = df.copy()
    def pct(x): 
        try: return f"{float(x)*100:.2f}%"
        except: return "—"
    with open(fp,"w",encoding="utf-8") as f:
        f.write("| family | Daily‑Long | Daily‑Short | Δ(D) | Weekly‑Long | Weekly‑Short | Δ(W) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _,r in df2.iterrows():
            f.write("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                r["family"], pct(r["daily_long"]), pct(r["daily_short"]), pct(r["daily_delta"]),
                pct(r["weekly_long"]), pct(r["weekly_short"]), pct(r["weekly_delta"])
            ))

def main():
    ap = argparse.ArgumentParser(description="Build compact κ table and run HAC tests for Long−Short.")
    ap.add_argument("--daily-summary", type=str, required=True, help="Daily long_short_summary.csv")
    ap.add_argument("--weekly-summary", type=str, required=True, help="Weekly long_short_summary.csv")
    ap.add_argument("--daily-ts-dir", type=str, required=True, help="Folder of daily per‑set long_short__*.csv")
    ap.add_argument("--weekly-ts-dir", type=str, required=True, help="Folder of weekly per‑set long_short__*.csv")
    ap.add_argument("--mapping", type=str, default="21:4,42:8,63:13,126:26,252:52", help="Daily→Weekly mapping")
    ap.add_argument("--freq-out", type=str, default="weekly", choices=["weekly","monthly"], help="Aggregation frequency for HAC tests.")
    ap.add_argument("--nw-lag", type=str, default="auto", help="NW lag (int or 'auto').")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    args = ap.parse_args()

    setup_logger("INFO")
    ensure_dir(args.out_dir)
    mapping = parse_mapping(args.mapping)

    # 讀 summary 產精簡表
    dsum = read_summary(args.daily_summary)
    wsum = read_summary(args.weekly_summary)
    table = compact_table_by_family(dsum, wsum, mapping)
    csv_path = os.path.join(args.out_dir, "ls_compact_by_family.csv")
    md_path  = os.path.join(args.out_dir, "ls_compact_by_family.md")
    table.to_csv(csv_path, index=False)
    write_md_table(table, md_path)
    print(f"[WRITE] {csv_path}")
    print(f"[WRITE] {md_path}")

    # 收集 per‑set 差序列並做 HAC
    dmap = collect_diff_series(args.daily_ts_dir,  mapping, "daily")
    wmap = collect_diff_series(args.weekly_ts_dir, mapping, "weekly")
    d_hac = pooled_hac_by_family(dmap, freq_out=args.freq_out, lag=args.nw_lag)
    w_hac = pooled_hac_by_family(wmap, freq_out=args.freq_out, lag=args.nw_lag)

    d_hac.to_csv(os.path.join(args.out_dir, "hac_tests_by_family_daily.csv"), index=False)
    w_hac.to_csv(os.path.join(args.out_dir, "hac_tests_by_family_weekly.csv"), index=False)
    print(f"[WRITE] {os.path.join(args.out_dir, 'hac_tests_by_family_daily.csv')}")
    print(f"[WRITE] {os.path.join(args.out_dir, 'hac_tests_by_family_weekly.csv')}")

    # overall
    T_d, mean_d, t_d, p_d = pooled_hac_overall(dmap, freq_out=args.freq_out, lag=args.nw_lag)
    T_w, mean_w, t_w, p_w = pooled_hac_overall(wmap, freq_out=args.freq_out, lag=args.nw_lag)
    with open(os.path.join(args.out_dir, "hac_tests_overall.txt"), "w", encoding="utf-8") as f:
        f.write("== Pooled HAC on (Long−Short) ==\n")
        f.write(f"Daily:  T={T_d}  mean_ann={mean_d:.4%}  t={t_d:.3f}  p(two‑sided)={p_d:.4f}\n")
        f.write(f"Weekly: T={T_w}  mean_ann={mean_w:.4%}  t={t_w:.3f}  p(two‑sided)={p_w:.4f}\n")
    print(f"[WRITE] {os.path.join(args.out_dir, 'hac_tests_overall.txt')}")
    print("Done.")

if __name__ == "__main__":
    main()