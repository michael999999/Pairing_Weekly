#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Walk-forward OOS selection & evaluation at Set level (L × W)
- 訓練：前 n 個年度（--train-periods），以年度 LOO 交叉驗證挑選 θ*（z_entry, z_exit, time_stop）
- 測試：用 θ* 僅在下一年度交易，記錄 OOS 日報酬與指標
- 聚合：每個 Set 串接所有 OOS，算 Sharpe、PSR（非正態修正），得 p-value
- FDR：對每個 z_window W，用 BH 程序控制 FDR 選最終 L（以 p-value 與 Sharpe 綜合）
- 螢幕：列出各年度（如 2015–2019）的 OOS θ 與績效，最後列累計績效

註解：繁體中文；print/log 英文
"""

import argparse
import json
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# 匯入週頻快取 Loader（不修改回測核心）
try:
    from .cache_loader import RollingCacheLoader
except Exception:
    from src.cache_loader import RollingCacheLoader


# ========= 小工具 =========

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_floats_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_ints_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_time_stop_grid(s: str) -> List[Optional[int]]:
    """grid 解析：支援 'none' 回傳 None 表示無時間停損（其餘為週數整數）。"""
    out: List[Optional[int]] = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if t in ("none", "nan", ""):
            out.append(None)
        else:
            out.append(int(float(t)))
    return out

def to_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    """保證 pair_id 存在（方向化，stock1__stock2）。"""
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
    """掃描 cache 根目錄取得可用的 L 與 Z。"""
    Ls = []
    Zs = set()
    if not root.exists():
        return Ls, sorted(list(Zs))
    for Ldir in root.iterdir():
        if Ldir.is_dir() and Ldir.name.upper().startswith("L"):
            try:
                Lval = float(int(Ldir.name[1:])) / 100.0
            except Exception:
                continue
            Ls.append(Lval)
            for Zdir in Ldir.iterdir():
                if Zdir.is_dir() and Zdir.name.upper().startswith("Z"):
                    try:
                        Zval = int(Zdir.name[1:])
                        Zs.add(Zval)
                    except Exception:
                        continue
    return sorted(Ls), sorted(list(Zs))

def ann_metrics(returns: pd.Series, freq: int = 252) -> Dict[str, float]:
    """年化指標（日頻年化）。"""
    r = returns.dropna()
    if len(r) == 0:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0)
    mu = r.mean() * freq
    vol = r.std(ddof=1) * sqrt(freq) if r.std(ddof=1) > 0 else 0.0
    sharpe = mu / vol if vol > 0 else 0.0
    return dict(ann_return=float(mu), ann_vol=float(vol), sharpe=float(sharpe))

def max_drawdown_curve(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0

def psr_prob(ret: pd.Series, sr0: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio（非正態修正；Lopez de Prado）
    - 使用「未年化」Sharpe：SR_hat = mu/sigma（以日頻）
    - z = sqrt(N-1) * (SR_hat - SR0) / sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2)
    - 返回 PSR = Phi(z)
    """
    r = ret.dropna()
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return np.nan
    sr_hat = r.mean() / r.std(ddof=1)   # 未年化 SR
    skew = float(r.skew())
    kurt_ex = float(r.kurt())           # excess kurtosis（常態為 0）
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr_hat + ((kurt_ex - 1.0) / 4.0) * (sr_hat ** 2)))
    z = np.sqrt(n - 1.0) * (sr_hat - sr0) / denom
    # 正態 CDF
    from math import erf, sqrt as msqrt
    psr = 0.5 * (1.0 + erf(z / msqrt(2.0)))
    return float(psr)

def bh_fdr(pvals: List[float], q: float = 0.1) -> Tuple[float, List[int]]:
    """
    Benjamini–Hochberg FDR 控制
    - 傳入 p 值清單，回傳臨界值與通過的索引（基於排序前的原始順序）
    """
    m = len(pvals)
    arr = np.asarray(pvals, dtype=float)
    order = np.argsort(arr)
    thr_index = -1
    thr_value = np.nan
    for rank, idx in enumerate(order, start=1):
        if arr[idx] <= (rank / m) * q:
            thr_index = idx
            thr_value = (rank / m) * q
    passed = []
    if thr_index >= 0:
        thr = arr[thr_index]
        for i, p in enumerate(arr):
            if p <= thr:
                passed.append(i)
        return thr, passed
    return np.nan, []


# ========= 部位生成與年面板 =========

def build_positions(z: pd.DataFrame, z_entry: float, z_exit: float, time_stop_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """由 z 生成部位與持有日數（外部已 shift，避免前視）。"""
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
    z_signal: pd.DataFrame   # [dates × pairs]（已 shift(1)）
    r_pair: pd.DataFrame     # [dates × pairs]
    beta_abs: pd.DataFrame   # [dates × pairs] 用於成本
    pair_ids: List[str]
    w_per_pair: float        # 等權權重（每對資金分配）

def prepare_year_panel(loader: RollingCacheLoader,
                       pair_ids: List[str],
                       tp_start: str,
                       tp_end: str,
                       price_type: str) -> Optional[YearPanel]:
    """讀入該年度所需矩陣，產出 YearPanel。"""
    date_range = (tp_start, tp_end)
    panel_z = loader.load_panel(pair_ids, fields=("z",), date_range=date_range, join="outer", allow_missing=True)
    panel_b = loader.load_panel(pair_ids, fields=("beta",), date_range=date_range, join="outer", allow_missing=True)
    panel_px = loader.load_panel(pair_ids, fields=("px",), date_range=date_range, join="outer", allow_missing=True)
    if panel_z.empty or panel_b.empty or panel_px.empty:
        return None

    # 對齊日期
    dates = panel_z.index.union(panel_b.index).union(panel_px.index).sort_values()
    z = panel_z.reindex(dates)["z"]
    beta = panel_b.reindex(dates)["beta"]

    # 價格欄位
    cols0 = panel_px.columns.get_level_values(0)
    if "px_x" in cols0 and "px_y" in cols0:
        xkey, ykey = "px_x", "px_y"
    elif "px_x_raw" in cols0 and "px_y_raw" in cols0:
        xkey, ykey = "px_x_raw", "px_y_raw"
    else:
        return None
    px = panel_px.reindex(dates)
    if price_type == "log":
        rx = px[xkey].diff()
        ry = px[ykey].diff()
    else:
        rx = px[xkey].pct_change()
        ry = px[ykey].pct_change()

    beta_lag = beta.shift(1)
    r_pair = (ry - beta_lag * rx)
    z_signal = z.shift(1)  # 不前視

    n_pairs = len(z.columns)
    if n_pairs == 0:
        return None
    w = 1.0 / float(n_pairs)

    return YearPanel(
        dates=dates,
        z_signal=z_signal,
        r_pair=r_pair,
        beta_abs=beta_lag.abs().fillna(0.0),
        pair_ids=list(z.columns),
        w_per_pair=w
    )

def eval_year(year_panel: YearPanel,
              z_entry: float,
              z_exit: float,
              time_stop_days: Optional[int],
              cost_bps: float,
              capital: float) -> Tuple[pd.Series, Dict[str, float], Dict[str, float], bool, List[float], List[int]]:
    """
    單年度評估：回傳
    - ret_series（日報酬，含成本）
    - metrics：年化、MDD
    - trade_stats（逐筆交易統計）：總交易數、勝率、平均持有天數、profit factor
    - no_trade_flag：Σ|Δpos|==0
    - trade_pnls：逐筆交易 PnL（含成本）
    - trade_durs：逐筆持有天數
    """
    zsig = year_panel.z_signal
    r_pair = year_panel.r_pair
    beta_abs = year_panel.beta_abs
    w = year_panel.w_per_pair
    cap = float(capital)

    tstop = time_stop_days if time_stop_days is not None else 10**9

    pos, days = build_positions(zsig, z_entry=z_entry, z_exit=z_exit, time_stop_days=tstop)

    pnl_ex = (pos.shift(1) * r_pair * (w * cap)).sum(axis=1)
    dpos = pos.fillna(0.0).diff().abs()
    traded_notional = ((w * cap) * (1.0 + beta_abs) * dpos).sum(axis=1)
    cost = traded_notional * (float(cost_bps) / 10000.0)

    pnl_net = pnl_ex - cost
    ret = pnl_net / cap
    equity = (1.0 + ret.fillna(0.0)).cumprod()
    dd = max_drawdown_curve(equity)

    m = ann_metrics(ret)
    m["max_drawdown"] = float(dd.min()) if len(dd) else 0.0

    # 逐筆交易統計
    trade_pnls: List[float] = []
    trade_durs: List[int] = []

    cost_rate = float(cost_bps) / 10000.0
    for pid in pos.columns:
        pos_s = pos[pid]
        r_s = r_pair[pid]
        beta_abs_s = beta_abs[pid]
        pnl_ex_s = (pos_s.shift(1) * r_s * (w * cap))
        cost_s = ((w * cap) * (1.0 + beta_abs_s) * pos_s.fillna(0.0).diff().abs()) * cost_rate
        pnl_net_s = pnl_ex_s - cost_s
        holding = pos_s.fillna(0.0) != 0.0
        if holding.any():
            h = holding.astype(int).values
            idx = np.where(np.diff(np.r_[0, h, 0]) != 0)[0]
            for j in range(0, len(idx), 2):
                start = idx[j]
                end = idx[j + 1] - 1
                if end >= start + 1:
                    pnl_trade = float(pnl_net_s.iloc[start + 1: end + 1].sum())
                    dur_days = int(end - start + 1)
                    trade_pnls.append(pnl_trade)
                    trade_durs.append(dur_days)

    total_trades = len(trade_pnls)
    sum_win = sum(p for p in trade_pnls if p > 0)
    sum_loss = sum(-p for p in trade_pnls if p < 0)
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = (wins / total_trades) if total_trades > 0 else np.nan
    profit_factor = (sum_win / sum_loss) if sum_loss > 0 else (np.inf if sum_win > 0 else np.nan)
    avg_duration_days = (np.mean(trade_durs) if trade_durs else np.nan)

    trade_stats = dict(
        total_trades=int(total_trades),
        win_rate=float(win_rate) if win_rate == win_rate else np.nan,
        avg_duration_days=float(avg_duration_days) if avg_duration_days == avg_duration_days else np.nan,
        profit_factor=float(profit_factor) if profit_factor not in (np.inf, -np.inf) else np.inf
    )

    no_trade_flag = bool(dpos.sum(axis=1).sum() == 0.0)

    return ret, m, trade_stats, no_trade_flag, trade_pnls, trade_durs


# ========= Walk-forward 主邏輯 =========

@dataclass
class OOSYearRow:
    year: str
    z_entry: float
    z_exit: float
    time_stop_weeks: Optional[int]
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    avg_duration_days: float
    profit_factor: float

@dataclass
class SetOOSResult:
    L: float
    Z: int
    years: List[str]
    oos_params_rows: List[OOSYearRow]
    ret_oos: pd.Series          # 串接日 OOS 報酬
    equity_oos: pd.Series
    metrics_oos: Dict[str, float]
    psr: float
    p_value: float
    set_dir: str

def select_theta_by_cv(train_years: List[str],
                       year_panels: Dict[str, YearPanel],
                       z_entry_grid: List[float],
                       z_exit_grid: List[float],
                       tstop_grid_weeks: List[Optional[int]],
                       cost_bps: float,
                       capital: float) -> Tuple[float, float, Optional[int]]:
    """
    訓練窗內以年度 LOO 當 CV：逐年當作 validation，選擇平均 Sharpe 最佳（平手看總報酬、再看年化波動）。
    回傳 θ* = (z_entry, z_exit, time_stop_weeks)
    """
    cand_scores = []
    for ze in z_entry_grid:
        for zx in z_exit_grid:
            for tsw in tstop_grid_weeks:
                time_stop_days = None if tsw is None else int(ceil(float(tsw) * 5.0))
                sh_list = []
                ret_list = []
                vol_list = []
                for y in train_years:
                    panel = year_panels.get(y)
                    if panel is None:
                        continue
                    ret_y, m_y, _, _, _, _ = eval_year(
                        year_panel=panel,
                        z_entry=float(ze),
                        z_exit=float(zx),
                        time_stop_days=time_stop_days,
                        cost_bps=float(cost_bps),
                        capital=float(capital)
                    )
                    sh_list.append(float(m_y["sharpe"]))
                    ret_list.append(float(ret_y.sum()))
                    vol_list.append(float(m_y["ann_vol"]))
                if not sh_list:
                    continue
                sh_avg = float(np.nanmean(sh_list))
                ret_tot = float(np.nansum(ret_list))
                vol_avg = float(np.nanmean(vol_list))
                cand_scores.append(dict(ze=ze, zx=zx, tsw=tsw, sh=sh_avg, ret=ret_tot, vol=vol_avg))
    if not cand_scores:
        # 預設回傳常用門檻
        return 2.0, 0.5, None

    # 依準則排序：Sharpe 降冪 → 總報酬降冪 → 年化波動升冪
    cand_sorted = sorted(cand_scores, key=lambda r: (round(r["sh"], 10), round(r["ret"], 10), -round(r["vol"], 10)), reverse=True)
    best = cand_sorted[0]
    return float(best["ze"]), float(best["zx"]), (int(best["tsw"]) if best["tsw"] is not None else None)


def run_set_walkforward(L: float,
                        Z: int,
                        sel_df: pd.DataFrame,
                        loader_root: str,
                        price_type: str,
                        all_years: List[str],
                        train_periods: int,
                        z_entry_grid: List[float],
                        z_exit_grid: List[float],
                        tstop_grid_weeks: List[Optional[int]],
                        cost_bps: float,
                        capital: float,
                        n_pairs_cap: int,
                        ignore_selection_formation: bool,
                        out_root: Path) -> Optional[SetOOSResult]:
    """單一 Set（L×Z）之走動式樣本外流程。"""
    set_dir = out_root / f"L{int(round(L*100)):03d}_Z{int(Z):03d}"
    ensure_dir(set_dir)

    loader = RollingCacheLoader(
        root=loader_root,
        price_type=price_type,
        formation_length=float(L),
        z_window=int(Z),
        log_level="ERROR"
    )

    # 預先建立每年的 YearPanel（加速重用）
    year_panels: Dict[str, YearPanel] = {}
    for y in all_years:
        if ignore_selection_formation:
            g = sel_df[sel_df["trading_period"].astype(str) == y].copy()
        else:
            g = sel_df[(sel_df["trading_period"].astype(str) == y) &
                       (pd.to_numeric(sel_df["formation_length"], errors="coerce") == float(L))].copy()
        if g.empty:
            year_panels[y] = None
            continue
        if "rank_final" in g.columns:
            g = g.sort_values(["rank_final"], ascending=True)
        pair_ids = g["pair_id"].dropna().astype(str).unique().tolist()[:int(n_pairs_cap)]
        if not pair_ids:
            year_panels[y] = None
            continue
        t_start = str(g["trading_start"].iloc[0]) if "trading_start" in g.columns else f"{y}-01-01"
        t_end   = str(g["trading_end"].iloc[0])   if "trading_end" in g.columns   else f"{y}-12-31"
        panel = prepare_year_panel(loader, pair_ids, t_start, t_end, price_type=price_type)
        year_panels[y] = panel

    # 走動式：從第 train_periods 年後開始 OOS
    oos_rows: List[OOSYearRow] = []
    all_oos_returns: List[pd.Series] = []

    for i in range(train_periods, len(all_years)):
        train_years = [y for y in all_years[i-train_periods:i] if year_panels.get(y) is not None]
        test_year = all_years[i]
        panel_test = year_panels.get(test_year)
        if not train_years or panel_test is None:
            # 跳過本次 OOS
            continue

        # 訓練窗內挑 θ*
        ze_star, zx_star, tsw_star = select_theta_by_cv(
            train_years=train_years,
            year_panels=year_panels,
            z_entry_grid=z_entry_grid,
            z_exit_grid=z_exit_grid,
            tstop_grid_weeks=tstop_grid_weeks,
            cost_bps=cost_bps,
            capital=capital
        )
        tstop_days = None if tsw_star is None else int(ceil(float(tsw_star) * 5.0))

        # 測試窗：以 θ* 交易
        ret_oos_y, m_y, tstats_y, _, _, _ = eval_year(
            year_panel=panel_test,
            z_entry=ze_star, z_exit=zx_star, time_stop_days=tstop_days,
            cost_bps=cost_bps, capital=capital
        )
        all_oos_returns.append(ret_oos_y)

        oos_rows.append(OOSYearRow(
            year=str(test_year),
            z_entry=float(ze_star),
            z_exit=float(zx_star),
            time_stop_weeks=(int(tsw_star) if tsw_star is not None else None),
            ann_return=float(m_y["ann_return"]),
            ann_vol=float(m_y["ann_vol"]),
            sharpe=float(m_y["sharpe"]),
            max_drawdown=float(m_y["max_drawdown"]),
            total_trades=int(tstats_y["total_trades"]),
            win_rate=float(tstats_y["win_rate"]) if tstats_y["win_rate"] == tstats_y["win_rate"] else np.nan,
            avg_duration_days=float(tstats_y["avg_duration_days"]) if tstats_y["avg_duration_days"] == tstats_y["avg_duration_days"] else np.nan,
            profit_factor=float(tstats_y["profit_factor"]) if tstats_y["profit_factor"] == tstats_y["profit_factor"] else np.nan
        ))

    if not oos_rows:
        return None

    # 聚合 OOS
    ret_oos = pd.concat(all_oos_returns).sort_index()
    equity_oos = (1.0 + ret_oos.fillna(0.0)).cumprod()
    dd_oos = max_drawdown_curve(equity_oos)
    m_full = ann_metrics(ret_oos)
    m_full["max_drawdown"] = float(dd_oos.min()) if len(dd_oos) else 0.0
    cum_return = float(equity_oos.iloc[-1] - 1.0) if len(equity_oos) else 0.0

    # PSR 與 p-value（對 SR0=0）
    psr = psr_prob(ret_oos, sr0=0.0)
    p_value = float(1.0 - psr) if psr == psr else np.nan

    # 儲存檔案
    # 1) 每年 OOS θ 與績效
    df_rows = pd.DataFrame([r.__dict__ for r in oos_rows])
    df_rows = df_rows.sort_values("year")
    df_rows.to_csv(set_dir / "wf_oos_params.csv", index=False, encoding="utf-8-sig")

    # 2) OOS 全期曲線與報酬
    eq_df = pd.DataFrame({"date": ret_oos.index, "ret": ret_oos.values})
    eq_df["equity"] = equity_oos.reindex(ret_oos.index).values
    eq_df.to_csv(set_dir / "wf_oos_returns.csv", index=False, encoding="utf-8-sig")

    # 3) 摘要
    summary = dict(
        formation_length=float(L),
        z_window=int(Z),
        cum_return=float(cum_return),
        ann_return=float(m_full["ann_return"]),
        ann_vol=float(m_full["ann_vol"]),
        sharpe=float(m_full["sharpe"]),
        max_drawdown=float(m_full["max_drawdown"]),
        psr=float(psr) if psr == psr else np.nan,
        p_value=float(p_value) if p_value == p_value else np.nan,
        years=[r.year for r in oos_rows]
    )
    with open(set_dir / "wf_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return SetOOSResult(
        L=float(L),
        Z=int(Z),
        years=[r.year for r in oos_rows],
        oos_params_rows=oos_rows,
        ret_oos=ret_oos,
        equity_oos=equity_oos,
        metrics_oos=dict(
            cum_return=cum_return,
            ann_return=float(m_full["ann_return"]),
            ann_vol=float(m_full["ann_vol"]),
            sharpe=float(m_full["sharpe"]),
            max_drawdown=float(m_full["max_drawdown"])
        ),
        psr=float(psr) if psr == psr else np.nan,
        p_value=float(p_value) if p_value == p_value else np.nan,
        set_dir=str(set_dir)
    )


# ========= 螢幕輸出（每年行 + 累計行） =========

def print_oos_table(set_name: str, rows: List[OOSYearRow], total_metrics: Dict[str, float]):
    """美化列印每年 OOS 與總計。"""
    headers = ["year","z_entry","z_exit","time_stop","ann_return","ann_vol","sharpe","max_drawdown",
               "total_trades","win_rate","avg_duration_days","profit_factor"]
    # 轉字串
    data = []
    for r in rows:
        def pct(x): return f"{x:.2%}" if x == x and np.isfinite(x) else "NA"
        def num(x, n=1): return f"{x:.{n}f}" if x == x and np.isfinite(x) else "NA"
        data.append([
            str(r.year),
            num(r.z_entry,1),
            num(r.z_exit,1),
            ("none" if r.time_stop_weeks is None else f"{r.time_stop_weeks}w"),
            pct(r.ann_return),
            pct(r.ann_vol),
            num(r.sharpe,2),
            pct(r.max_drawdown),
            f"{r.total_trades}",
            pct(r.win_rate),
            num(r.avg_duration_days,1),
            ("inf" if not np.isfinite(r.profit_factor) else num(r.profit_factor,2))
        ])
    # 欄寬
    widths = []
    for j, h in enumerate(headers):
        w = len(h)
        for i in range(len(data)):
            w = max(w, len(data[i][j]))
        widths.append(w + 1)
    # 列印
    print(f"\nOOS per-year ({set_name}):")
    header = "".join(h.ljust(widths[j]) for j, h in enumerate(headers)).rstrip()
    print(header)
    print("-" * len(header))
    for row in data:
        line = "".join(row[j].rjust(widths[j]) if headers[j] not in ("year","time_stop") else row[j].ljust(widths[j]) for j in range(len(headers))).rstrip()
        print(line)
    # 總計
    def pct(x): return f"{x:.2%}" if x == x and np.isfinite(x) else "NA"
    def num(x, n=2): return f"{x:.{n}f}" if x == x and np.isfinite(x) else "NA"
    total_line = f"Total OOS -> Sharpe={num(total_metrics['sharpe'],2)} AnnRet={pct(total_metrics['ann_return'])} AnnVol={pct(total_metrics['ann_vol'])} MDD={pct(total_metrics['max_drawdown'])}"
    print(total_line)


# ========= 入口 =========

def main():
    ap = argparse.ArgumentParser(description="Walk-forward OOS selection/evaluation per Set (L×W) using CV on training window.")
    ap.add_argument("--top-csv", type=str, default="cache/top_pairs_annual.csv", help="Selection CSV.")
    ap.add_argument("--cache-root", type=str, default="cache/rolling_cache_weekly_v1", help="Weekly cache root.")
    ap.add_argument("--price-type", type=str, default="log", choices=["log","raw"], help="Price type.")
    ap.add_argument("--formation-lengths", type=str, default="all", help='Comma floats in years or "all".')
    ap.add_argument("--z-windows", type=str, default="all", help='Comma ints in weeks or "all".')
    ap.add_argument("--trading-periods", type=str, default="all", help='Comma years or "all".')
    ap.add_argument("--train-periods", type=int, default=3, help="Number of preceding periods used for training.")
    ap.add_argument("--grid-z-entry", type=str, default="0.5,1.0,1.5,2.0,2.5", help="Grid for z_entry.")
    ap.add_argument("--grid-z-exit", type=str, default="0.0,0.5", help="Grid for z_exit.")
    ap.add_argument("--grid-time-stop", type=str, default="none,6", help='Grid for time stop (weeks); supports "none".')
    ap.add_argument("--cost-bps", type=float, default=5.0, help="Per-leg, one-way cost in bps.")
    ap.add_argument("--capital", type=float, default=1_000_000.0, help="Capital for scaling.")
    ap.add_argument("--n-pairs-cap", type=int, default=60, help="Max pairs per year.")
    ap.add_argument("--ignore-selection-formation", action="store_true", help="Reuse same pair list across L.")
    ap.add_argument("--n-jobs", type=int, default=8, help="Parallel jobs across Sets.")
    ap.add_argument("--backend", type=str, default="loky", choices=["loky","threading"], help="Joblib backend.")
    ap.add_argument("--fdr-q", type=float, default=0.1, help="BH FDR q-level per z_window.")
    ap.add_argument("--out-dir", type=str, default="reports/walkforward_weekly", help="Output root.")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    summary_dir = out_root / "_summary"
    ensure_dir(summary_dir)

    # 讀 selection
    print(f"[INFO] Loading selection: {args.top_csv}")
    sel = pd.read_csv(args.top_csv, encoding="utf-8-sig", low_memory=False)
    sel = to_pair_id(sel)

    # 年度清單
    if args.trading_periods.lower() == "all":
        years_all = sorted(sel["trading_period"].astype(str).unique().tolist())
    else:
        years_all = [x.strip() for x in args.trading_periods.split(",") if x.strip()]
    if len(years_all) <= args.train_periods:
        print(f"[ERROR] Not enough periods: total={len(years_all)} <= train_periods={args.train_periods}")
        return

    # L 與 Z
    if args.formation_lengths.lower() == "all" or args.z_windows.lower() == "all":
        Ls_avail, Zs_avail = list_LZ_from_cache(Path(args.cache_root))
    else:
        Ls_avail, Zs_avail = [], []
    L_list = Ls_avail if args.formation_lengths.lower() == "all" else parse_floats_list(args.formation_lengths)
    Z_list = Zs_avail if args.z_windows.lower() == "all" else parse_ints_list(args.z_windows)
    if not L_list or not Z_list:
        print(f"[ERROR] No L or Z found. L={L_list}, Z={Z_list}")
        return

    # Grid
    z_entry_grid = parse_floats_list(args.grid_z_entry)
    z_exit_grid = parse_floats_list(args.grid_z_exit)
    tstop_grid_weeks = parse_time_stop_grid(args.grid_time_stop)

    print(f"[INFO] WF Sets: L={L_list} × Z={Z_list} | train_periods={args.train_periods} | years={years_all}")
    print(f"[INFO] Grid: z_entry={z_entry_grid} z_exit={z_exit_grid} time_stop={tstop_grid_weeks}")

    # 平行化：各 Set 執行
    tasks = [(L, Z) for L in L_list for Z in Z_list]

    def run_one(L: float, Z: int) -> Optional[SetOOSResult]:
        return run_set_walkforward(
            L=L, Z=Z,
            sel_df=sel,
            loader_root=args.cache_root,
            price_type=args.price_type,
            all_years=years_all,
            train_periods=int(args.train_periods),
            z_entry_grid=z_entry_grid,
            z_exit_grid=z_exit_grid,
            tstop_grid_weeks=tstop_grid_weeks,
            cost_bps=float(args.cost_bps),
            capital=float(args.capital),
            n_pairs_cap=int(args.n_pairs_cap),
            ignore_selection_formation=bool(args.ignore_selection_formation),
            out_root=out_root
        )

    results_raw = Parallel(n_jobs=int(args.n_jobs), backend=args.backend)(
        delayed(run_one)(L, Z) for (L, Z) in tasks
    )
    results: List[SetOOSResult] = [r for r in results_raw if r is not None]

    if not results:
        print("[ERROR] No WF results produced.")
        return

    # 保存各 Set 摘要
    rows = []
    for r in results:
        rows.append(dict(
            formation_length=float(r.L),
            z_window=int(r.Z),
            cum_return=float(r.metrics_oos["cum_return"]),
            ann_return=float(r.metrics_oos["ann_return"]),
            ann_vol=float(r.metrics_oos["ann_vol"]),
            sharpe=float(r.metrics_oos["sharpe"]),
            max_drawdown=float(r.metrics_oos["max_drawdown"]),
            psr=float(r.psr) if r.psr == r.psr else np.nan,
            p_value=float(r.p_value) if r.p_value == r.p_value else np.nan,
            years=",".join(r.years),
            set_dir=str(r.set_dir)
        ))
    df_sets = pd.DataFrame(rows).sort_values(["z_window","formation_length"]).reset_index(drop=True)
    df_sets.to_csv(summary_dir / "wf_sets_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {summary_dir / 'wf_sets_summary.csv'}")

    # 每個 W 用 BH FDR 選 L
    picks = []
    for Z in sorted(df_sets["z_window"].unique().tolist()):
        dfW = df_sets[df_sets["z_window"] == Z].copy()
        # p 值取 PSR 對 SR0=0 的 p = 1 - PSR
        pvals = dfW["p_value"].astype(float).tolist()
        thr, passed_idx = bh_fdr(pvals, q=float(args.fdr_q))
        if passed_idx:
            df_pass = dfW.iloc[passed_idx].sort_values(["sharpe","cum_return","ann_vol"], ascending=[False,False,True])
            best_row = df_pass.iloc[0]
            reason = f"BH-FDR q={args.fdr_q} passed (thr≈{thr:.4f})"
        else:
            # 無通過者：取最小 p 或 Sharpe 最大
            dfW = dfW.sort_values(["p_value","sharpe","cum_return","ann_vol"], ascending=[True,False,False,True])
            best_row = dfW.iloc[0]
            reason = "no pass; pick min p (then max Sharpe)"
        picks.append(dict(
            z_window=int(Z),
            formation_length=float(best_row["formation_length"]),
            sharpe=float(best_row["sharpe"]),
            ann_return=float(best_row["ann_return"]),
            ann_vol=float(best_row["ann_vol"]),
            max_drawdown=float(best_row["max_drawdown"]),
            psr=float(best_row["psr"]) if best_row["psr"] == best_row["psr"] else np.nan,
            p_value=float(best_row["p_value"]) if best_row["p_value"] == best_row["p_value"] else np.nan,
            set_dir=str(best_row["set_dir"]),
            select_reason=reason
        ))
    df_picks = pd.DataFrame(picks).sort_values("z_window")
    df_picks.to_csv(summary_dir / "wf_final_picks_by_W.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {summary_dir / 'wf_final_picks_by_W.csv'}")

    # 螢幕列印：對每個 W 的最終 L，列出 2015–2019（或你樣本）每年的 OOS θ 與績效，最後印累計
    for _, pick in df_picks.iterrows():
        set_dir = Path(str(pick["set_dir"]))
        L = float(pick["formation_length"]); Z = int(pick["z_window"])
        # 讀每年 OOS
        df_oos = pd.read_csv(set_dir / "wf_oos_params.csv", encoding="utf-8-sig")
        df_oos = df_oos.sort_values("year")
        rows = []
        for _, r in df_oos.iterrows():
            rows.append(OOSYearRow(
                year=str(r["year"]),
                z_entry=float(r["z_entry"]),
                z_exit=float(r["z_exit"]),
                time_stop_weeks=(int(r["time_stop_weeks"]) if pd.notna(r["time_stop_weeks"]) else None),
                ann_return=float(r["ann_return"]),
                ann_vol=float(r["ann_vol"]),
                sharpe=float(r["sharpe"]),
                max_drawdown=float(r["max_drawdown"]),
                total_trades=int(r["total_trades"]),
                win_rate=float(r["win_rate"]) if pd.notna(r["win_rate"]) else np.nan,
                avg_duration_days=float(r["avg_duration_days"]) if pd.notna(r["avg_duration_days"]) else np.nan,
                profit_factor=float(r["profit_factor"]) if pd.notna(r["profit_factor"]) else np.nan
            ))
        # 讀總計
        with open(set_dir / "wf_summary.json", "r", encoding="utf-8") as f:
            s = json.load(f)
        metrics = dict(
            ann_return=float(s["ann_return"]),
            ann_vol=float(s["ann_vol"]),
            sharpe=float(s["sharpe"]),
            max_drawdown=float(s["max_drawdown"])
        )
        print_oos_table(set_name=f"L={L}, Z={Z}", rows=rows, total_metrics=metrics)

    print(f"\n[INFO] Walk-forward finished. Root: {out_root.resolve()}")


if __name__ == "__main__":
    main()