#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
timeseries_tests.py (enhanced)
- 讀取日頻與週頻的 OOS 回報時間序列（ts_daily.csv, ts_weekly.csv），
  以 (formation_length L, family=日↔週等效視窗) 為配對鍵，將兩邊回報匯總成同一頻率（預設：月頻），
  在「同風險（target volatility）」下進行以下檢定：
  1) Newey–West（HAC）成對均值檢定：Δreturn@TV（W−D）
  2) Jobson–Korkie with Memmel correction（近似）：ΔSharpe（W−D）
  3) Stationary/Block Bootstrap：Δreturn@TV 與 ΔSharpe 的 95% CI 與 p 值
  4) 多重比較：BH FDR（各指標分開計算 q 值）
  5) 近顯著配對 Top-K：各檢定依 p 值排序輸出前 K 名（含 CI/Δ/T）
  6) 橫斷面聚合（pooled）NW 檢定：先對每期的 Δreturn@TV 做聚合，再跑 NW 檢定
     - --pooled-weight {equal,invvar}
     - --pooled-bootstrap（對 pooled Δ 做 bootstrap，輸出 pooled_bootstrap.json）
- 註解：繁體中文；print/log：英文
"""

import os
import sys
import argparse
import logging
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd


# ====== 基本工具 ======

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
        return ""


# ====== 回報匯總與年化 ======

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


def aggregate_returns(r: pd.Series, freq_out: str = "monthly") -> pd.Series:
    """
    將日或週回報匯總為指定頻率：
    - monthly：每月連乘 − 1（使用月末 'ME'）
    - weekly：以 W-FRI 為週末，連乘 − 1
    """
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return r
    if freq_out.lower().startswith("m"):
        out = r.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    else:
        out = r.resample("W-FRI").apply(lambda x: (1.0 + x).prod() - 1.0)
    return out.dropna()


def annualization_k(freq_out: str) -> int:
    return 12 if freq_out.lower().startswith("m") else 52


def ann_metrics(r: pd.Series, k: int) -> Tuple[float, float, float]:
    """
    年化均值、年化波動、年化 Sharpe（0 無風險）。
    - k = 12（月頻）或 52（週頻）
    """
    rr = pd.Series(r).dropna()
    if len(rr) < 2:
        return 0.0, 0.0, 0.0
    mu = rr.mean() * k
    sd = rr.std(ddof=1) * np.sqrt(k)
    sr = (mu / sd) if sd > 0 else 0.0
    return float(mu), float(sd), float(sr)


# ====== Newey–West（HAC）t 檢定 ======

def _autocov(x: np.ndarray, lag: int) -> float:
    n = len(x)
    if lag >= n:
        return 0.0
    x0 = x - x.mean()
    return float(np.dot(x0[lag:], x0[:n - lag]) / n)


def nw_hac_ttest(diff: pd.Series, lag: Optional[int] = None) -> Tuple[float, float, int, float]:
    """
    Newey–West（HAC）對「均值」的 t 檢定（雙尾）：
    - 輸入 diff：Δr（W−D），同頻率且對齊
    - lag：None/'auto' 自動帶（floor(4*(T/100)^(2/9) )）
    回傳：t 值、p 值、T、se（HAC 標準誤）
    """
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


# ====== Jobson–Korkie with Memmel correction（近似） ======

def jkm_sharpe_test(r1: pd.Series, r2: pd.Series, k: int) -> Tuple[float, float, float, float]:
    """
    Memmel 修正的 Jobson–Korkie 檢定（近似實作）：
    - r1, r2：同頻回報（未年化）
    - k：年化因子（12 或 52）
    回傳：z 值、p 值、SR1_ann、SR2_ann
    """
    x = pd.Series(r1).dropna()
    y = pd.Series(r2).dropna()
    idx = x.index.intersection(y.index)
    x = x.reindex(idx).dropna()
    y = y.reindex(idx).dropna()
    T = int(min(len(x), len(y)))
    if T < 6:
        return 0.0, 1.0, 0.0, 0.0

    mx, sx = float(x.mean()), float(x.std(ddof=1))
    my, sy = float(y.mean()), float(y.std(ddof=1))
    if sx <= 0 or sy <= 0:
        return 0.0, 1.0, 0.0, 0.0

    sr1 = mx / sx
    sr2 = my / sy
    rho = float(np.corrcoef(x, y)[0, 1]) if T > 2 else 0.0

    var_jk = (1.0 / T) * (2.0 * (1.0 - rho) + 0.5 * (sr1 ** 2 + sr2 ** 2) - rho * sr1 * sr2)
    var_jkm = max(1e-12, var_jk * (1.0 - 0.5 * sr1 ** 2 - 0.5 * sr2 ** 2))
    z = float((sr2 - sr1) / np.sqrt(var_jkm))

    sr1_ann = float(sr1 * np.sqrt(k))
    sr2_ann = float(sr2 * np.sqrt(k))

    from math import erf, sqrt
    cdf = 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))
    p = float(2.0 * (1.0 - cdf))
    return z, p, sr1_ann, sr2_ann


# ====== Stationary Bootstrap ======

def stationary_bootstrap_indices(T: int, B: int, block_len: int, rng: np.random.Generator) -> List[np.ndarray]:
    p = 1.0 / max(1, int(block_len))
    idx_list: List[np.ndarray] = []
    for _ in range(B):
        out = np.empty(T, dtype=int)
        out[0] = int(rng.integers(0, T))
        for t in range(1, T):
            if rng.random() < p:
                out[t] = int(rng.integers(0, T))
            else:
                out[t] = (out[t - 1] + 1) % T
        idx_list.append(out)
    return idx_list


def bootstrap_ci_pvalue(x: pd.Series, y: pd.Series, k: int, B: int = 5000, block_len: int = 3,
                        use_target_vol: bool = True, target_vol: float = 0.10,
                        seed: int = 42) -> Dict[str, float]:
    """
    Stationary bootstrap（同步抽樣）估計 Δreturn@TV 與 ΔSharpe 的 CI 與 p（雙尾）。
    - x, y：同頻回報（未年化）
    - 若 use_target_vol=True：依 k（12/52）估年化波動後將兩者靜態縮放至 target_vol
    """
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    idx = x.index.intersection(y.index)
    x = x.reindex(idx).dropna()
    y = y.reindex(idx).dropna()
    T = int(min(len(x), len(y)))
    if T < 6:
        return dict(
            bs_ret_ci_lo=np.nan, bs_ret_ci_hi=np.nan, bs_ret_p=np.nan,
            bs_sh_ci_lo=np.nan, bs_sh_ci_hi=np.nan, bs_sh_p=np.nan
        )

    # 同風險靜態縮放
    if use_target_vol:
        volx = float(x.std(ddof=1) * np.sqrt(k))
        voly = float(y.std(ddof=1) * np.sqrt(k))
        sx = (target_vol / volx) if volx > 0 else 1.0
        sy = (target_vol / voly) if voly > 0 else 1.0
    else:
        sx = sy = 1.0
    xd = x * sx
    yd = y * sy

    # 觀測統計量（年化）
    dx = float(xd.mean()) * k
    dy = float(yd.mean()) * k
    dret = dy - dx

    sdx = float(xd.std(ddof=1))
    sdy = float(yd.std(ddof=1))
    sr_x = (float(xd.mean()) / sdx) * np.sqrt(k) if sdx > 0 else 0.0
    sr_y = (float(yd.mean()) / sdy) * np.sqrt(k) if sdy > 0 else 0.0
    dsh = sr_y - sr_x

    # Bootstrap
    rng = np.random.default_rng(seed)
    idx_sets = stationary_bootstrap_indices(T, int(B), int(block_len), rng)
    dret_b = []
    dsh_b = []
    for I in idx_sets:
        xx = xd.values[I]
        yy = yd.values[I]
        dret_b.append((np.mean(yy) - np.mean(xx)) * k)
        sxx = float(np.std(xx, ddof=1))
        syy = float(np.std(yy, ddof=1))
        sr_xb = (np.mean(xx) / sxx) * np.sqrt(k) if sxx > 0 else 0.0
        sr_yb = (np.mean(yy) / syy) * np.sqrt(k) if syy > 0 else 0.0
        dsh_b.append(sr_yb - sr_xb)

    dret_b = np.array(dret_b, dtype=float)
    dsh_b = np.array(dsh_b, dtype=float)

    def _pval(obs: float, draws: np.ndarray) -> float:
        cnt = float(np.sum(np.abs(draws) >= abs(obs)))
        return float((cnt + 1.0) / (len(draws) + 1.0))

    ret_ci_lo, ret_ci_hi = np.percentile(dret_b, [2.5, 97.5])
    sh_ci_lo,  sh_ci_hi  = np.percentile(dsh_b,  [2.5, 97.5])
    p_ret = _pval(dret, dret_b)
    p_sh  = _pval(dsh,  dsh_b)

    return dict(
        bs_ret_ci_lo=float(ret_ci_lo), bs_ret_ci_hi=float(ret_ci_hi), bs_ret_p=float(p_ret),
        bs_sh_ci_lo=float(sh_ci_lo),   bs_sh_ci_hi=float(sh_ci_hi),   bs_sh_p=float(p_sh)
    )


# ====== FDR（Benjamini–Hochberg） ======

def fdr_bh(pvals: List[float]) -> List[float]:
    p = np.array(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        qi = p[order[i]] * n / ranks[i]
        qi = min(qi, prev)
        q[i] = qi
        prev = qi
    out = np.empty(n, dtype=float)
    out[order] = q
    return out.tolist()


# ====== 主流程 ======

def main():
    ap = argparse.ArgumentParser(description="Timeseries tests: NW/JK-M/Bootstrap on matched daily vs weekly sets.")
    ap.add_argument("--ts-daily", type=str, required=True, help="Path to ts_daily.csv")
    ap.add_argument("--ts-weekly", type=str, required=True, help="Path to ts_weekly.csv")
    ap.add_argument("--freq-out", type=str, default="monthly", choices=["monthly","weekly"], help="Aggregation frequency for tests.")
    ap.add_argument("--target-vol", type=float, default=0.10, help="Static target volatility for risk scaling (0=disabled).")
    ap.add_argument("--nw-lag", type=str, default="auto", help="Newey–West lag (int or 'auto').")
    ap.add_argument("--bootstrap-b", type=int, default=5000, help="Bootstrap resamples.")
    ap.add_argument("--block-len", type=int, default=3, help="Expected block length for stationary bootstrap (e.g., 3 for monthly, 8~12 for weekly).")
    ap.add_argument("--min-samples", type=int, default=24, help="Minimum aligned samples (periods) to run tests.")
    ap.add_argument("--top-k", type=int, default=10, help="Top-K nearest-to-significance rows to export per test.")
    ap.add_argument("--pooled-average", action="store_true", help="Also run pooled NW test on cross-sectional average Δreturn@TV.")
    ap.add_argument("--pooled-weight", type=str, default="equal", choices=["equal","invvar"], help="Cross-sectional weight for pooled average.")
    ap.add_argument("--pooled-bootstrap", action="store_true", help="Run stationary bootstrap for pooled Δ and save pooled_bootstrap.json.")
    ap.add_argument("--out-dir", type=str, default="reports/summary/tests", help="Output directory.")
    args = ap.parse_args()

    setup_logger("INFO")
    ensure_dir(args.out_dir)

    # 讀日/週時間序列
    if not os.path.isfile(args.ts_daily) or not os.path.isfile(args.ts_weekly):
        print("[ERROR] timeseries CSV not found."); sys.exit(2)

    d = pd.read_csv(args.ts_daily, encoding="utf-8-sig", low_memory=False)
    w = pd.read_csv(args.ts_weekly, encoding="utf-8-sig", low_memory=False)

    # 欄名統一小寫
    d.columns = [c.strip().lower() for c in d.columns]
    w.columns = [c.strip().lower() for c in w.columns]

    # 必要欄位
    need_d = {"date","l","daily_z","ret","family"}
    need_w = {"date","l","weekly_z","ret","family"}
    if not need_d.issubset(set(d.columns)):
        print(f"[ERROR] ts_daily.csv needs columns: {sorted(list(need_d))}"); sys.exit(2)
    if not need_w.issubset(set(w.columns)):
        print(f"[ERROR] ts_weekly.csv needs columns: {sorted(list(need_w))}"); sys.exit(2)

    # 轉 DatetimeIndex
    d = to_datetime_index(d, "date")
    w = to_datetime_index(w, "date")

    # 依 (L,family) 匯總
    key_d = d.groupby(["l","family","daily_z"])
    key_w = w.groupby(["l","family","weekly_z"])

    # 建立配對鍵（L,family）
    pairs_d = {(float(L), str(fam)): int(dz) for (L, fam, dz) in key_d.groups.keys()}
    pairs_w = {(float(L), str(fam)): int(wz) for (L, fam, wz) in key_w.groups.keys()}
    keys = sorted(list(set(pairs_d.keys()).intersection(set(pairs_w.keys()))))

    if not keys:
        print("[ERROR] No matched (L,family) keys between daily and weekly."); sys.exit(2)

    freq_out = args.freq_out.lower()
    k = annualization_k(freq_out)
    lag = None if str(args.nw_lag).lower() == "auto" else int(args.nw_lag)
    target_vol = float(args.target_vol)
    use_tv = (target_vol > 0.0)

    rows: List[Dict[str, float]] = []
    skipped = 0

    # 儲存每配對的 Δreturn@TV 時序（供 pooled）
    diff_series_panel: List[pd.Series] = []
    diff_series_keys: List[Tuple[float,str,int,int]] = []  # (L,family,dz,wz)

    for (L, fam) in keys:
        dz = pairs_d[(L, fam)]
        wz = pairs_w[(L, fam)]

        rd = d[(d["l"] == L) & (d["family"] == fam) & (d["daily_z"] == dz)]["ret"]
        rw = w[(w["l"] == L) & (w["family"] == fam) & (w["weekly_z"] == wz)]["ret"]
        if len(rd) == 0 or len(rw) == 0:
            skipped += 1; continue

        rd_f = aggregate_returns(rd, freq_out=freq_out)
        rw_f = aggregate_returns(rw, freq_out=freq_out)

        idx = rd_f.index.intersection(rw_f.index)
        rd_f = rd_f.reindex(idx).dropna()
        rw_f = rw_f.reindex(idx).dropna()
        T = int(min(len(rd_f), len(rw_f)))
        if T < int(args.min_samples):
            skipped += 1; continue

        if use_tv:
            vold = float(rd_f.std(ddof=1) * np.sqrt(k))
            volw = float(rw_f.std(ddof=1) * np.sqrt(k))
            sd = (target_vol / vold) if vold > 0 else 1.0
            sw = (target_vol / volw) if volw > 0 else 1.0
        else:
            sd = sw = 1.0

        rd_s = rd_f * sd
        rw_s = rw_f * sw

        mu_d, vol_d, sr_d = ann_metrics(rd_s, k=k)
        mu_w, vol_w, sr_w = ann_metrics(rw_s, k=k)

        diff = (rw_s - rd_s)

        # 存入 pooled 用
        diff_series_panel.append(diff.copy())
        diff_series_keys.append((L, fam, dz, wz))

        # Pair-level tests
        t_nw, p_nw, T_nw, se_nw = nw_hac_ttest(diff, lag=lag if lag is not None else "auto")
        z_jkm, p_jkm, sr_d_ann, sr_w_ann = jkm_sharpe_test(rd_s, rw_s, k=k)
        bs = bootstrap_ci_pvalue(rd_f, rw_f, k=k, B=int(args.bootstrap_b), block_len=int(args.block_len),
                                 use_target_vol=use_tv, target_vol=target_vol, seed=42)

        rows.append(dict(
            L=float(L),
            family=str(fam),
            daily_z=int(dz),
            weekly_z=int(wz),
            freq_out=str(freq_out),
            target_vol=float(target_vol),
            T_aligned=int(T),

            ann_return_d=float(mu_d),
            ann_vol_d=float(vol_d),
            sharpe_d=float(sr_d),
            ann_return_w=float(mu_w),
            ann_vol_w=float(vol_w),
            sharpe_w=float(sr_w),

            delta_ann_return=float(mu_w - mu_d),
            delta_sharpe=float(sr_w - sr_d),

            nw_t=float(t_nw),
            nw_p=float(p_nw),
            nw_se=float(se_nw),

            jkm_z=float(z_jkm),
            jkm_p=float(p_jkm),

            bs_ret_ci_lo=float(bs["bs_ret_ci_lo"]),
            bs_ret_ci_hi=float(bs["bs_ret_ci_hi"]),
            bs_ret_p=float(bs["bs_ret_p"]),
            bs_sh_ci_lo=float(bs["bs_sh_ci_lo"]),
            bs_sh_ci_hi=float(bs["bs_sh_ci_hi"]),
            bs_sh_p=float(bs["bs_sh_p"])
        ))

    if not rows:
        print("[ERROR] No valid pairs for tests. Skipped=", skipped)
        sys.exit(2)

    res = pd.DataFrame(rows).sort_values(["L","daily_z","weekly_z"]).reset_index(drop=True)

    # FDR（分別對 NW 與 JK-M、Bootstrap 做 BH）
    for col_p, col_q in [("nw_p","nw_q"), ("jkm_p","jkm_q"), ("bs_ret_p","bs_ret_q"), ("bs_sh_p","bs_sh_q")]:
        plist = res[col_p].fillna(1.0).tolist()
        qlist = fdr_bh(plist)
        res[col_q] = qlist

    out_csv = os.path.join(args.out_dir, "tests_results.csv")
    res.to_csv(out_csv, index=False)
    print(f"[WRITE] {out_csv}")

    # ---- Top-K 近顯著配對輸出（NW / JK-M / Bootstrap-Return / Bootstrap-Sharpe）----
    topk = max(1, int(args.top_k))
    def dump_topk(col_p: str, fname: str, extra_cols: List[str]):
        df = res.sort_values(col_p, ascending=True).head(topk).copy()
        cols = ["L","family","daily_z","weekly_z","T_aligned","delta_ann_return","delta_sharpe",
                "nw_t","nw_p","jkm_z","jkm_p","bs_ret_p","bs_sh_p",
                "bs_ret_ci_lo","bs_ret_ci_hi","bs_sh_ci_lo","bs_sh_ci_hi"] + extra_cols
        cols = [c for c in cols if c in df.columns]
        fp = os.path.join(args.out_dir, fname)
        df[cols].to_csv(fp, index=False)
        print(f"[WRITE] {fp}")

    dump_topk("nw_p",      "near_hits_nw.csv",      ["nw_q"])
    dump_topk("jkm_p",     "near_hits_jkm.csv",     ["jkm_q"])
    dump_topk("bs_ret_p",  "near_hits_bs_ret.csv",  ["bs_ret_q"])
    dump_topk("bs_sh_p",   "near_hits_bs_sh.csv",   ["bs_sh_q"])

    # ---- 橫斷面聚合（pooled）----
    if bool(args.pooled_average) and len(diff_series_panel) > 0:
        # 組 panel（index union）
        panel_df = pd.concat(diff_series_panel, axis=1)
        panel_df.columns = [f"L{float(L):.1f}_{fam}_Dz{dz}_Wz{wz}" for (L, fam, dz, wz) in diff_series_keys]

        # 等權
        pooled_equal = panel_df.mean(axis=1, skipna=True).dropna()
        # 輸出（等權）
        pd.DataFrame({"date": pooled_equal.index, "pooled_diff": pooled_equal.values}).to_csv(
            os.path.join(args.out_dir, "pooled_diff_series_equal.csv"), index=False
        )
        # 保留舊名以兼容既有繪圖腳本
        pd.DataFrame({"date": pooled_equal.index, "pooled_diff": pooled_equal.values}).to_csv(
            os.path.join(args.out_dir, "pooled_diff_series.csv"), index=False
        )

        t_eq, p_eq, T_eq, se_eq = nw_hac_ttest(pooled_equal, lag=lag if lag is not None else "auto")
        mu_eq_ann = float(pooled_equal.mean()) * k

        # 逆變異數加權（若選 invvar）
        pooled_inv = None
        if str(args.pooled_weight).lower() == "invvar":
            # 以全期方差的倒數為權重；每期依可用樣本重新正規化
            var_col = panel_df.var(axis=0, ddof=1).replace([0.0, np.inf, -np.inf], np.nan)
            w0 = 1.0 / var_col
            w0 = w0.fillna(0.0)

            # 權重向量套用到每期（僅針對非 NaN col 正規化）
            num = panel_df.mul(w0, axis=1).sum(axis=1, skipna=True)
            den = panel_df.notna().mul(w0, axis=1).sum(axis=1, skipna=True).replace(0.0, np.nan)
            pooled_inv = (num / den).dropna()

            pd.DataFrame({"date": pooled_inv.index, "pooled_diff": pooled_inv.values}).to_csv(
                os.path.join(args.out_dir, "pooled_diff_series_invvar.csv"), index=False
            )

        # 文字摘要
        lines = []
        lines.append("Pooled average Δreturn@TV across pairs (NW):")
        lines.append(f"- equal: T={T_eq} | mean={fmt_pct(mu_eq_ann)} | t={t_eq:.3f} | p={p_eq:.4f} | se(HAC)={se_eq:.6f}")
        if pooled_inv is not None and len(pooled_inv) > 3:
            t_iv, p_iv, T_iv, se_iv = nw_hac_ttest(pooled_inv, lag=lag if lag is not None else "auto")
            mu_iv_ann = float(pooled_inv.mean()) * k
            lines.append(f"- invvar: T={T_iv} | mean={fmt_pct(mu_iv_ann)} | t={t_iv:.3f} | p={p_iv:.4f} | se(HAC)={se_iv:.6f}")

        # Bootstrap（pooled）
        if bool(args.pooled_bootstrap):
            out_bs = {}
            # helper
            def _pooled_bs(s: pd.Series) -> Dict[str, float]:
                x = pd.Series(s).dropna().values
                T = len(x)
                if T < 6: 
                    return {"ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "T": T}
                rng = np.random.default_rng(123)
                idx_sets = stationary_bootstrap_indices(T, int(args.bootstrap_b), int(args.block_len), rng)
                draws = []
                for I in idx_sets:
                    draws.append(np.mean(x[I]) * k)
                draws = np.array(draws)
                mu = float(np.mean(x) * k)
                ci_lo, ci_hi = np.percentile(draws, [2.5, 97.5])
                p = float((np.sum(np.abs(draws) >= abs(mu)) + 1.0) / (len(draws) + 1.0))
                return {"mean_ann": mu, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "p": p, "T": T}

            out_bs["equal"] = _pooled_bs(pooled_equal)
            if pooled_inv is not None and len(pooled_inv) > 5:
                out_bs["invvar"] = _pooled_bs(pooled_inv)

            import json
            with open(os.path.join(args.out_dir, "pooled_bootstrap.json"), "w", encoding="utf-8") as f:
                json.dump(out_bs, f, indent=2)
            print(f"[WRITE] {os.path.join(args.out_dir, 'pooled_bootstrap.json')}")

        # 附加到 summary.txt
        out_txt = os.path.join(args.out_dir, "tests_summary.txt")
        with open(out_txt, "a", encoding="utf-8") as f:
            f.write("\n\n" + "\n".join(lines))
        print("\n".join(lines))

    # ---- Summary（整體）----
    def _sig_rate(col_p: str, alpha: float = 0.05) -> float:
        p = pd.to_numeric(res[col_p], errors="coerce").dropna()
        return float((p < alpha).mean()) if len(p) else np.nan

    summary_lines = []
    summary_lines.append(f"=== Timeseries Tests Summary ===")
    summary_lines.append(f"Pairs tested: {len(res)} | Skipped pairs: {skipped}")
    summary_lines.append(f"Freq out: {args.freq_out} | target_vol: {fmt_pct(args.target_vol)} | min_samples: {args.min_samples}")
    summary_lines.append("")
    summary_lines.append(f"Mean Δreturn@TV (W−D): {fmt_pct(res['delta_ann_return'].mean())}")
    summary_lines.append(f"Mean ΔSharpe (W−D): {res['delta_sharpe'].mean():.3f}")
    summary_lines.append("")
    summary_lines.append("Significance (raw p<0.05):")
    summary_lines.append(f"- NW (Δreturn@TV): {fmt_pct(_sig_rate('nw_p'))}")
    summary_lines.append(f"- JK-M (ΔSharpe): {fmt_pct(_sig_rate('jkm_p'))}")
    summary_lines.append(f"- Bootstrap (Δreturn@TV): {fmt_pct(_sig_rate('bs_ret_p'))}")
    summary_lines.append(f"- Bootstrap (ΔSharpe): {fmt_pct(_sig_rate('bs_sh_p'))}")

    out_txt = os.path.join(args.out_dir, "tests_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"[WRITE] {out_txt}")
    print("\n".join(summary_lines))
    print("Done.")

if __name__ == "__main__":
    main()