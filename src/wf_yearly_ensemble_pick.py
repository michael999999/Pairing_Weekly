#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wf_yearly_ensemble_pick.py
- 年度走動式（walk-forward）「集成 OOS」：
  每個訓練窗（前 n 年）在所有 L×Z×參數格內，以 PSR→Sharpe→CumRet→AnnVol 排序，
  選出前 M 名 (L×Z×θ)，於測試年以等權或 PSR 權重疊加其日報酬，得到該年的 OOS 報酬。
- 圖表：年度收益柱狀（策略 vs 基準）、全期累積（策略 vs 基準；x 軸只顯示年份）
- 不改核心語義：T+1 收盤成交（PnL 用 pos.shift(1) × r），beta 用 beta.shift(1)；成本 per-leg bps。
- 註解：繁體中文；print/log 英文
"""

import argparse
import json
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# ===== 匯入 Loader =====
try:
    from .cache_loader import RollingCacheLoader
except Exception:
    from src.cache_loader import RollingCacheLoader


# ===== 通用工具 =====

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_tstop(s: str) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for t in s.split(","):
        tt = t.strip().lower()
        if tt in ("none", "", "nan"):
            out.append(None)
        else:
            out.append(int(float(tt)))
    return out

def to_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    if "pair_id" in df.columns:
        return df
    if "stock1" in df.columns and "stock2" in df.columns:
        df["pair_id"] = df["stock1"].astype(str) + "__" + df["stock2"].astype(str)
        return df
    if "pair" in df.columns:
        df["pair_id"] = df["pair"].astype(str)
        return df
    raise KeyError("Selection CSV needs 'pair_id' or ('stock1','stock2') columns.")

def list_LZ_from_cache(root: Path) -> Tuple[List[float], List[int]]:
    Ls = []
    Zs = set()
    if not root.exists():
        return Ls, sorted(list(Zs))
    for Ld in root.iterdir():
        if Ld.is_dir() and Ld.name.upper().startswith("L"):
            try:
                Lval = float(int(Ld.name[1:])) / 100.0
            except Exception:
                continue
            Ls.append(Lval)
            for Zd in Ld.iterdir():
                if Zd.is_dir() and Zd.name.upper().startswith("Z"):
                    try:
                        Zs.add(int(Zd.name[1:]))
                    except Exception:
                        continue
    return sorted(Ls), sorted(list(Zs))

def ann_metrics(returns: pd.Series, freq: int = 252) -> Dict[str, float]:
    r = returns.dropna()
    if len(r) == 0:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0)
    mu = r.mean() * freq
    sd = r.std(ddof=1)
    vol = sd * sqrt(freq) if sd > 0 else 0.0
    sharpe = mu / vol if vol > 0 else 0.0
    return dict(ann_return=float(mu), ann_vol=float(vol), sharpe=float(sharpe))

def max_drawdown_curve(eq: pd.Series) -> pd.Series:
    peak = eq.cummax()
    return eq / peak - 1.0

def psr_prob(r: pd.Series, sr0: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio（Lopez de Prado 非正態修正）"""
    x = r.dropna()
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return np.nan
    sr_hat = x.mean() / x.std(ddof=1)  # 未年化 SR
    skew = float(x.skew())
    kurt_ex = float(x.kurt())
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr_hat + ((kurt_ex - 1.0) / 4.0) * (sr_hat ** 2)))
    z = np.sqrt(n - 1.0) * (sr_hat - sr0) / denom
    from math import erf, sqrt as msqrt
    return float(0.5 * (1.0 + erf(z / msqrt(2.0))))


# ===== 回測基本件（與核心一致） =====

def build_positions(z: pd.DataFrame, z_entry: float, z_exit: float, time_stop_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos = z.copy() * np.nan
    days = z.copy() * 0.0
    for pid in z.columns:
        z_ser = z[pid]
        pos_ser = pd.Series(index=z.index, dtype="float64")
        dcount = pd.Series(index=z.index, dtype="float64")
        last_pos = 0.0
        hold_days = 0
        for i, _ in enumerate(z.index):
            sig = z_ser.iloc[i]
            curr = last_pos
            if curr == 0:
                if pd.notna(sig) and sig >= z_entry:
                    curr = -1.0
                    hold_days = 1
                elif pd.notna(sig) and sig <= -z_entry:
                    curr = +1.0
                    hold_days = 1
                else:
                    curr = 0.0
                    hold_days = 0
            else:
                exit_flag = (pd.notna(sig) and abs(sig) <= z_exit) or (hold_days >= time_stop_days)
                if exit_flag:
                    curr = 0.0
                    hold_days = 0
                else:
                    hold_days += 1
            pos_ser.iloc[i] = curr
            dcount.iloc[i] = hold_days
            last_pos = curr
        pos[pid] = pos_ser
        days[pid] = dcount
    return pos, days

@dataclass
class YearPanel:
    dates: pd.DatetimeIndex
    z_signal: pd.DataFrame
    r_pair: pd.DataFrame
    beta_abs: pd.DataFrame
    pair_ids: List[str]
    w_per_pair: float

def prepare_year_panel(loader: RollingCacheLoader,
                       pair_ids: List[str],
                       start: str, end: str,
                       price_type: str) -> Optional[YearPanel]:
    date_range = (start, end)
    panel_z  = loader.load_panel(pair_ids, fields=("z",),  date_range=date_range, join="outer", allow_missing=True)
    panel_b  = loader.load_panel(pair_ids, fields=("beta",),date_range=date_range, join="outer", allow_missing=True)
    panel_px = loader.load_panel(pair_ids, fields=("px",),  date_range=date_range, join="outer", allow_missing=True)
    if panel_z.empty or panel_b.empty or panel_px.empty:
        return None
    dates = panel_z.index.union(panel_b.index).union(panel_px.index).sort_values()
    z = panel_z.reindex(dates)["z"]
    beta = panel_b.reindex(dates)["beta"]

    cols0 = panel_px.columns.get_level_values(0)
    if "px_x" in cols0 and "px_y" in cols0:
        xkey, ykey = "px_x", "px_y"
    elif "px_x_raw" in cols0 and "px_y_raw" in cols0:
        xkey, ykey = "px_x_raw", "px_y_raw"
    else:
        return None

    px = panel_px.reindex(dates)
    if price_type == "log":
        rx = px[xkey].diff(); ry = px[ykey].diff()
    else:
        rx = px[xkey].pct_change(); ry = px[ykey].pct_change()

    beta_lag = beta.shift(1)
    r_pair = (ry - beta_lag * rx)
    z_signal = z.shift(1)

    n = len(z.columns)
    if n == 0:
        return None
    w = 1.0 / float(n)
    return YearPanel(dates=dates, z_signal=z_signal, r_pair=r_pair,
                     beta_abs=beta_lag.abs().fillna(0.0),
                     pair_ids=list(z.columns), w_per_pair=w)

def eval_year(panel: YearPanel,
              ze: float, zx: float, tstop_days: Optional[int],
              cost_bps: float, capital: float) -> Tuple[pd.Series, Dict[str,float], Dict[str,float]]:
    zsig = panel.z_signal
    r_pair = panel.r_pair
    beta_abs = panel.beta_abs
    w = panel.w_per_pair
    cap = float(capital)
    tstop = tstop_days if tstop_days is not None else 10**9

    pos, days = build_positions(zsig, z_entry=ze, z_exit=zx, time_stop_days=tstop)
    pnl_ex = (pos.shift(1) * r_pair * (w * cap)).sum(axis=1)
    dpos = pos.fillna(0.0).diff().abs()
    traded_notional = ((w * cap) * (1.0 + beta_abs) * dpos).sum(axis=1)
    cost = traded_notional * (float(cost_bps) / 10000.0)
    pnl_net = pnl_ex - cost
    ret = pnl_net / cap

    eq = (1.0 + ret.fillna(0.0)).cumprod()
    dd = max_drawdown_curve(eq)
    m = ann_metrics(ret); m["max_drawdown"] = float(dd.min()) if len(dd) else 0.0

    # 交易統計（簡版，供排序用可忽略；此處不返回逐筆）
    total_trades = int((dpos.sum(axis=1) > 0).sum())  # event days（不作為輸出）
    tstats = dict(total_trades=total_trades)
    return ret, m, tstats


# ===== 走動式集成 OOS 主程式 =====

def main():
    ap = argparse.ArgumentParser(description="Walk-forward yearly ENSEMBLE OOS (top-M over L×Z×θ in training window).")
    # 資料與空間
    ap.add_argument("--top-csv", type=str, default="cache/top_pairs_annual.csv")
    ap.add_argument("--cache-root", type=str, default="cache/rolling_cache_weekly_v1")
    ap.add_argument("--price-type", type=str, default="log", choices=["log","raw"])
    ap.add_argument("--formation-lengths", type=str, default="all")
    ap.add_argument("--z-windows", type=str, default="all")
    ap.add_argument("--trading-periods", type=str, default="all")
    # WF 與網格
    ap.add_argument("--train-periods", type=int, default=1)
    ap.add_argument("--grid-z-entry", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--grid-z-exit", type=str, default="0.0,0.5,1.0,1.5,2.0")
    ap.add_argument("--grid-time-stop", type=str, default="none,6,9,12")
    # 成本與名單
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--n-pairs-cap", type=int, default=60)
    ap.add_argument("--ignore-selection-formation", action="store_true")
    # 集成控制
    ap.add_argument("--ensemble-top", type=int, default=3, help="Top M combos in training window.")
    ap.add_argument("--ensemble-weight", type=str, default="equal", choices=["equal","psr"], help="Weighting scheme.")
    # 平行化
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--backend", type=str, default="loky", choices=["loky","threading"])
    # 輸出
    ap.add_argument("--out-dir", type=str, default="reports/wf_yearly_ensemble_pick")
    # 圖表
    ap.add_argument("--benchmark-symbol", type=str, default="_GSPC")
    ap.add_argument("--prices", type=str, default="data/prices.pkl")

    args = ap.parse_args()

    out_root = Path(args.out_dir); ensure_dir(out_root)

    # 讀 selection 與年度序列
    sel = pd.read_csv(args.top_csv, encoding="utf-8-sig", low_memory=False)
    sel = to_pair_id(sel)
    years_all = (sorted(sel["trading_period"].astype(str).unique().tolist())
                 if args.trading_periods.lower()=="all" else
                 [x.strip() for x in args.trading_periods.split(",") if x.strip()])
    if len(years_all) <= args.train_periods:
        print(f"[ERROR] Not enough periods: total={len(years_all)} <= train_periods={args.train_periods}")
        return

    # L/Z 空間
    if args.formation_lengths.lower()=="all" or args.z_windows.lower()=="all":
        Ls_avail, Zs_avail = list_LZ_from_cache(Path(args.cache_root))
    else:
        Ls_avail, Zs_avail = [], []
    L_list = Ls_avail if args.formation_lengths.lower()=="all" else parse_floats(args.formation_lengths)
    Z_list = Zs_avail if args.z_windows.lower()=="all" else parse_ints(args.z_windows)
    if not L_list or not Z_list:
        print(f"[ERROR] No L or Z found. L={L_list}, Z={Z_list}")
        return

    # 參數格
    z_entry_grid = parse_floats(args.grid_z_entry)
    z_exit_grid  = parse_floats(args.grid_z_exit)
    tstop_grid   = parse_tstop(args.grid_time_stop)

    print(f"[INFO] WF ENSEMBLE years={years_all} train_periods={args.train_periods}")
    print(f"[INFO] Space: L={L_list} × Z={Z_list} × z_entry={z_entry_grid} × z_exit={z_exit_grid} × tstop={tstop_grid}")
    print(f"[INFO] Ensemble: top={args.ensemble_top} weight={args.ensemble_weight} | n_jobs={args.n_jobs} backend={args.backend}")

    # 輔助：年度 pair 清單
    def get_pairs_for_year(year: str, L: float) -> List[str]:
        if args.ignore_selection_formation:
            g = sel[sel["trading_period"].astype(str) == year].copy()
        else:
            g = sel[(sel["trading_period"].astype(str) == year) &
                    (pd.to_numeric(sel["formation_length"], errors="coerce") == float(L))].copy()
        if g.empty:
            return []
        if "rank_final" in g.columns:
            g = g.sort_values(["rank_final"], ascending=True)
        return g["pair_id"].dropna().astype(str).unique().tolist()[:int(args.n_pairs_cap)]

    # Worker：訓練窗內評分某 L×Z×θ（回傳 score 與 θ）
    def score_one_combo(L: float, Z: int, ze: float, zx: float, tsw: Optional[int], train_years: List[str]):
        loader = RollingCacheLoader(root=args.cache_root, price_type=args.price_type,
                                    formation_length=float(L), z_window=int(Z), log_level="ERROR")
        rets = []
        for y in train_years:
            pids = get_pairs_for_year(y, L)
            if not pids:
                continue
            panel = prepare_year_panel(loader, pids, f"{y}-01-01", f"{y}-12-31", price_type=args.price_type)
            if panel is None:
                continue
            t_days = None if tsw is None else int(ceil(float(tsw)*5.0))
            ret_y, m_y, _ = eval_year(panel, ze, zx, t_days, float(args.cost_bps), float(args.capital))
            rets.append(ret_y)
        if not rets:
            return None
        r_full = pd.concat(rets).sort_index()
        psr = psr_prob(r_full, sr0=0.0)
        m = ann_metrics(r_full)
        eq = (1.0 + r_full.fillna(0.0)).cumprod()
        cum_ret = float(eq.iloc[-1]-1.0) if len(eq) else 0.0
        score = ( (psr if psr==psr else -np.inf),
                  round(m["sharpe"],10),
                  round(cum_ret,10),
                  -round(m["ann_vol"],10) )
        return dict(L=float(L), Z=int(Z), ze=float(ze), zx=float(zx), tsw=(int(tsw) if tsw is not None else None),
                    score=score, psr=float(psr) if psr==psr else np.nan)

    # 輔助：測試年回測某組合，拿日報酬
    def oos_ret_for_combo(L: float, Z: int, ze: float, zx: float, tsw: Optional[int], year: str) -> Optional[pd.Series]:
        pids = get_pairs_for_year(year, L)
        if not pids:
            return None
        loader = RollingCacheLoader(root=args.cache_root, price_type=args.price_type,
                                    formation_length=float(L), z_window=int(Z), log_level="ERROR")
        panel = prepare_year_panel(loader, pids, f"{year}-01-01", f"{year}-12-31", price_type=args.price_type)
        if panel is None:
            return None
        t_days = None if tsw is None else int(ceil(float(tsw)*5.0))
        ret_y, _, _ = eval_year(panel, ze, zx, t_days, float(args.cost_bps), float(args.capital))
        return ret_y

    # 逐年走動
    rows = []
    all_oos = []

    for i in range(args.train_periods, len(years_all)):
        train_years = years_all[i-args.train_periods:i]
        test_year   = years_all[i]
        print(f"[INFO] Window train={train_years} -> test={test_year}")

        # 1) 訓練窗並行評分所有 L×Z×θ
        tasks = []
        for L in L_list:
            for Z in Z_list:
                for ze in z_entry_grid:
                    for zx in z_exit_grid:
                        for tsw in tstop_grid:
                            tasks.append((L,Z,ze,zx,tsw))
        results = Parallel(n_jobs=int(args.n_jobs), backend=args.backend)(
            delayed(score_one_combo)(L,Z,ze,zx,tsw,train_years) for (L,Z,ze,zx,tsw) in tasks
        )
        results = [r for r in results if r is not None]
        if not results:
            print(f"[WARN] No valid combos in train window; skip {test_year}")
            continue

        # 2) 取前 M 名（PSR→Sharpe→CumRet→AnnVol）
        results_sorted = sorted(results, key=lambda d: d["score"], reverse=True)
        topM = results_sorted[:int(args.ensemble_top)]

        # 權重：等權或 PSR 權重（負 PSR 視為 0）
        if args.ensemble_weight == "psr":
            ws = np.array([max(0.0, float(d["psr"])) for d in topM], dtype=float)
            if ws.sum() == 0:
                w = np.ones(len(topM)) / len(topM)
            else:
                w = ws / ws.sum()
        else:
            w = np.ones(len(topM)) / len(topM)

        # 3) 測試年：集成日報酬
        rets = []
        for k, d in enumerate(topM):
            r = oos_ret_for_combo(d["L"], d["Z"], d["ze"], d["zx"], d["tsw"], test_year)
            if r is not None:
                rets.append(w[k] * r.reindex(r.index))  # 權重乘上日報酬
        if not rets:
            print(f"[WARN] No OOS returns for {test_year}")
            continue
        ret_ens = pd.concat(rets, axis=1).sum(axis=1)
        all_oos.append(ret_ens)

        # 4) 年度指標
        m = ann_metrics(ret_ens)
        eq = (1.0 + ret_ens.fillna(0.0)).cumprod()
        m["max_drawdown"] = float(max_drawdown_curve(eq).min()) if len(eq) else 0.0

        # 紀錄本年所選組合（簡要字串）
        combos_str = ";".join([f"L{int(round(d['L']*100))}_Z{int(d['Z'])}_e{d['ze']}_x{d['zx']}_t{('none' if d['tsw'] is None else d['tsw'])}" for d in topM])

        rows.append(dict(
            year=str(test_year),
            ann_return=float(m["ann_return"]),
            ann_vol=float(m["ann_vol"]),
            sharpe=float(m["sharpe"]),
            max_drawdown=float(m["max_drawdown"]),
            selected_topM=combos_str
        ))

    if not rows:
        print("[ERROR] No OOS rows produced.")
        return

    # 5) 全期 OOS 指標與輸出
    ret_full = pd.concat(all_oos).sort_index()
    eq_full = (1.0 + ret_full.fillna(0.0)).cumprod()
    dd_full = max_drawdown_curve(eq_full)
    m_full = ann_metrics(ret_full)
    m_full["max_drawdown"] = float(dd_full.min()) if len(dd_full) else 0.0

    # 螢幕輸出（簡表）
    print("\nWF yearly ENSEMBLE (top-M) OOS summary by year:")
    print("year ann_return ann_vol sharpe max_drawdown")
    for r in rows:
        print(f"{r['year']:>4} {r['ann_return']:.2%} {r['ann_vol']:.2%} {r['sharpe']:.2f} {r['max_drawdown']:.2%}")
    print(f"Total OOS -> Sharpe={m_full['sharpe']:.2f} AnnRet={m_full['ann_return']:.2%} AnnVol={m_full['ann_vol']:.2%} MDD={m_full['max_drawdown']:.2%}")

    # 寫檔
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "ensemble_oos_yearly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": ret_full.index, "ret": ret_full.values, "equity": eq_full.reindex(ret_full.index).values}).to_csv(
        out_root / "ensemble_oos_returns.csv", index=False, encoding="utf-8-sig"
    )
    with open(out_root / "ensemble_oos_summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(ann_return=float(m_full["ann_return"]), ann_vol=float(m_full["ann_vol"]),
                       sharpe=float(m_full["sharpe"]), max_drawdown=float(m_full["max_drawdown"])), f, indent=2)

    # 6) 對比圖（策略 vs 基準）
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # 基準：對齊日期
        px = pd.read_pickle(args.prices)
        if args.benchmark_symbol not in px.columns:
            print(f"[WARN] Benchmark {args.benchmark_symbol} not found in prices. Skip plots.")
            return
        s = px[args.benchmark_symbol].dropna()
        s.index = pd.to_datetime(s.index)
        s = s.reindex(pd.to_datetime(ret_full.index)).ffill()
        bench_eq = s / s.iloc[0]
        bench_ret = s.pct_change().fillna(0.0)

        # 年度收益柱狀
        years = sorted({d.year for d in pd.to_datetime(ret_full.index)})
        def year_ret_from_daily(ret: pd.Series, year: int) -> float:
            r = ret[(ret.index.year == year)]
            return float((1.0 + r).prod() - 1.0) if len(r) else np.nan

        data = []
        for y in years:
            data.append(dict(
                year=str(y),
                strat_ret=year_ret_from_daily(ret_full, y),
                bench_ret=year_ret_from_daily(bench_ret, y)
            ))
        ydf = pd.DataFrame(data)

        # 畫年度收益
        x = np.arange(len(ydf))
        width = 0.38
        plt.figure(figsize=(max(6, 0.7*len(ydf)+2), 4.2))
        plt.bar(x - width/2, ydf["strat_ret"].values, width=width, label="Strategy", color="#1f77b4")
        plt.bar(x + width/2, ydf["bench_ret"].values, width=width, label=f"{args.benchmark_symbol}", color="#ff7f0e")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.xticks(x, ydf["year"].tolist(), rotation=0)
        plt.ylabel("Annual Return")
        plt.title("Yearly Returns (OOS) - Ensemble")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_root / "ensemble_yearly_returns_vs_benchmark.png", dpi=160)
        plt.close()

        # 全期累積
        dates = pd.to_datetime(ret_full.index)
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        ax.plot(dates, eq_full.values, label="Strategy (Ensemble)", color="#1f77b4")
        ax.plot(dates, bench_eq.values, label=f"{args.benchmark_symbol}", color="#ff7f0e", alpha=0.85)
        ax.set_ylabel("Cumulative Equity (norm.)")
        ax.set_title("Cumulative Equity (OOS) - Ensemble")
        ax.legend()
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlim(dates.min(), dates.max())
        plt.tight_layout()
        plt.savefig(out_root / "ensemble_cumulative_vs_benchmark.png", dpi=160)
        plt.close()

        print(f"[INFO] Figures saved under: {out_root.resolve()}")

    except Exception as e:
        print(f"[WARN] Plotting skipped: {e}")


if __name__ == "__main__":
    main()