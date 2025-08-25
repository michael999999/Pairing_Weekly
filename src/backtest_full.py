#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_full.py
- 多組合（formation_length × trading_period）之配對交易回測（支援週頻快取）
- 維持語義：stock1 = X（自變數）、stock2 = Y（因變數），pair_id = "stock1__stock2"
- 註解：繁體中文；print/log（日誌）：英文

主要改動（對齊 T+1 收盤成交）：
- PnL 使用 pos.shift(1) × r_pair（成交當天不計 PnL，從隔日開始）
- 新增 --time-stop-weeks（週數），於主程式換算為交易日數，覆蓋 --time-stop
- 價格欄位自動偵測（px_x/px_y 或 px_x_raw/px_y_raw），搭配 price_type 使用 diff 或 pct_change
"""

import os
import sys
import json
import math
import argparse
import logging
import ast
from math import ceil
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# 匯入快取讀取器（支援封包或直接腳本兩種啟動方式）
try:
    from .cache_loader import RollingCacheLoader
except Exception:
    from src.cache_loader import RollingCacheLoader


# ====== 工具 ======

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def setup_logger(level="INFO"):
    """初始化日誌（英文）。"""
    logger = logging.getLogger()
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

def to_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    """保證 pair_id 欄位存在（方向化：stock1__stock2）；若缺則由 pair 解析。"""
    if "stock1" in df.columns and "stock2" in df.columns:
        df["pair_id"] = df["stock1"].astype(str) + "__" + df["stock2"].astype(str)
        return df
    if "pair" in df.columns:
        def _to_tuple(s):
            try:
                # 使用安全的字面解析，避免執行任意字串
                x = ast.literal_eval(str(s))
                if isinstance(x, (list, tuple)) and len(x) == 2:
                    return str(x[0]), str(x[1])
            except Exception:
                pass
            return None, None
        tups = df["pair"].apply(_to_tuple)
        df["stock1"] = tups.apply(lambda t: t[0])
        df["stock2"] = tups.apply(lambda t: t[1])
        df["pair_id"] = df["stock1"].astype(str) + "__" + df["stock2"].astype(str)
        return df
    raise ValueError("Cannot derive pair_id: need 'stock1'/'stock2' or 'pair' column in top_pairs.csv")

def ann_metrics(returns: pd.Series, rf: float = 0.0, freq: int = 252) -> dict:
    """年化指標（簡化版）。"""
    r = returns.dropna()
    if len(r) == 0:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0)
    mu = r.mean() * freq
    vol = r.std(ddof=1) * math.sqrt(freq)
    sharpe = (mu - rf) / vol if vol > 0 else 0.0
    return dict(ann_return=float(mu), ann_vol=float(vol), sharpe=float(sharpe))

def max_drawdown_curve(equity: pd.Series) -> pd.Series:
    """回傳逐日回撤序列（相對歷史高點）。"""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd

def load_noise_gate_json(path: str) -> Dict[Tuple[float, str], dict]:
    """
    載入 period_noise_gate.json
    - key: (formation_length, trading_period) -> gate dict
    - 常用欄位：rec_topk, capital_scale, z_entry_add_sigma, crossing_eps_add, time_stop_scale, regime, fail_period
    """
    if (not path) or (not os.path.isfile(path)):
        logging.warning(f"Gate JSON not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gate: Dict[Tuple[float, str], dict] = {}
    for r in data:
        try:
            L = float(r.get("formation_length"))
            tp = str(r.get("trading_period"))
        except Exception:
            continue
        gate[(L, tp)] = r
    logging.info(f"Loaded gate JSON combos: {len(gate)} from {path}")
    return gate


# ====== 訊號與持倉邏輯 ======

def build_positions(z: pd.DataFrame,
                    z_entry: float,
                    z_exit: float,
                    time_stop_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    由 z 生成部位與持有日數（不前視：回測外部會 shift）。
    - 規則：|z| >= z_entry 進場；|z| <= z_exit 出場；持有日數達 time_stop_days 亦出場
    - 回傳：pos, days_in_trade（與 z 同形狀；pos ∈ {-1,0,+1}）
    """
    pos = z.copy() * np.nan
    days = z.copy() * 0.0

    for pid in z.columns:
        z_ser = z[pid]
        pos_ser = pd.Series(index=z.index, dtype="float64")
        dcount = pd.Series(index=z.index, dtype="float64")
        last_pos = 0.0
        hold_days = 0

        for i, dt in enumerate(z.index):
            # 使用 t-1 的訊號（由外部 shift 保證），這裡直接用 z_ser.iloc[i] 當當期信號
            sig = z_ser.iloc[i]

            curr = last_pos
            if curr == 0:
                if pd.notna(sig) and sig >= z_entry:
                    curr = -1.0  # short spread（做空 Y − βX）
                    hold_days = 1
                elif pd.notna(sig) and sig <= -z_entry:
                    curr = +1.0  # long spread
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


# ====== 單一 combo 回測 ======

def run_combo_backtest(pair_ids: List[str],
                       trading_start: str,
                       trading_end: str,
                       loader: RollingCacheLoader,
                       base_z_entry: float,
                       base_z_exit: float,
                       base_time_stop: int,
                       gate_cfg: Optional[dict],
                       cost_bps: float,
                       capital: float,
                       out_dir: str,
                       combo_tag: str) -> Tuple[pd.Series, pd.Series, dict, pd.DataFrame]:
    """
    回測單一 combo（固定 L 與一個 trading_period 的交易窗）。
    - 讀取 z, beta, px；以 t-1 決策，t 實現
    - 成本：雙腿單邊 bps × |Δpos| × 名目金額
    - 等權配置：w_pair = 1/N；未持倉權重閒置
    - 回傳：ret_series（投組日回報）、equity（淨值）、metrics、pair_stats
    """
    # 日期窗
    date_range = (trading_start, trading_end)

    # 讀取 panel
    panel_z  = loader.load_panel(pair_ids, fields=("z",),  date_range=date_range, join="outer", allow_missing=True)
    panel_b  = loader.load_panel(pair_ids, fields=("beta",), date_range=date_range, join="outer", allow_missing=True)
    panel_px = loader.load_panel(pair_ids, fields=("px",),  date_range=date_range, join="outer", allow_missing=True)
    if panel_z.empty or panel_b.empty or panel_px.empty:
        logging.warning(f"[{combo_tag}] Panel empty after loading. Skipped.")
        return pd.Series(dtype="float64"), pd.Series(dtype="float64"), {}, pd.DataFrame()

    # 對齊日期
    dates = panel_z.index.union(panel_b.index).union(panel_px.index)
    panel_z = panel_z.reindex(dates)
    panel_b = panel_b.reindex(dates)
    panel_px = panel_px.reindex(dates)

    # gate 調整（若提供）
    ze_add = float(gate_cfg.get("z_entry_add_sigma", 0.0)) if gate_cfg else 0.0
    cx_add = float(gate_cfg.get("crossing_eps_add", 0.0)) if gate_cfg else 0.0
    ts_scale = float(gate_cfg.get("time_stop_scale", 1.0)) if gate_cfg else 1.0
    z_entry = float(base_z_entry) + ze_add
    z_exit  = float(base_z_exit) + cx_add
    tstop   = max(1, int(round(float(base_time_stop) * ts_scale)))

    # 不前視：部位使用 t-1 的 z 生成 t 當期部位；當期部位吃「下一日」報酬（T+1 收盤）
    z_signal = panel_z["z"].shift(1)
    pos, days = build_positions(z_signal, z_entry=z_entry, z_exit=z_exit, time_stop_days=tstop)

    # β、價格、報酬
    beta = panel_b["beta"].copy()
    # 自動偵測價格欄位（log：px_x/px_y；raw：px_x_raw/px_y_raw）
    fields_lvl0 = panel_px.columns.get_level_values(0)
    if "px_x" in fields_lvl0 and "px_y" in fields_lvl0:
        xkey, ykey = "px_x", "px_y"
    elif "px_x_raw" in fields_lvl0 and "px_y_raw" in fields_lvl0:
        xkey, ykey = "px_x_raw", "px_y_raw"
    else:
        raise KeyError("Panel PX missing expected fields: px_x/px_y or px_x_raw/px_y_raw")

    if loader.price_type == "log":
        rx = panel_px[xkey].diff()
        ry = panel_px[ykey].diff()
    else:
        rx = panel_px[xkey].pct_change()
        ry = panel_px[ykey].pct_change()

    beta_lag = beta.shift(1)

    # 對沖組合報酬（r_pair）與等權
    r_pair = (ry - beta_lag * rx)
    n_pairs = len(pos.columns)
    if n_pairs == 0:
        logging.warning(f"[{combo_tag}] No valid pairs after filtering. Skipped.")
        return pd.Series(dtype="float64"), pd.Series(dtype="float64"), {}, pd.DataFrame()

    w = 1.0 / float(n_pairs)
    cap = float(capital)

    # T+1 收盤成交：PnL 使用「前一日部位」乘以「當日報酬」
    pnl_ex_cost = (pos.shift(1) * r_pair * (w * cap)).sum(axis=1)

    # 成本：|Δpos| × 名目（Y 腿 + |β|×X 腿）
    dpos = pos.fillna(0.0).diff().abs()
    beta_for_cost = beta_lag.abs().fillna(0.0)
    traded_notional = ((w * cap) * (1.0 + beta_for_cost) * dpos).sum(axis=1)
    cost = traded_notional * (float(cost_bps) / 10000.0)

    pnl_net = pnl_ex_cost - cost
    ret = pnl_net / cap
    equity = (1.0 + ret.fillna(0.0)).cumprod()

    # 指標與每對統計
    dd = max_drawdown_curve(equity)
    m = ann_metrics(ret)
    m["max_drawdown"] = float(dd.min()) if len(dd) else 0.0
    m["avg_active_pairs"] = float((pos.abs() > 0).sum(axis=1).mean())
    m["avg_daily_turnover"] = float(traded_notional.div(cap).mean())
    m["z_entry"] = float(z_entry)
    m["z_exit"] = float(z_exit)
    m["time_stop_days"] = int(tstop)
    m["n_pairs"] = int(n_pairs)

    trade_counts = (dpos > 0).sum(axis=0)
    avg_hold = days.replace(0, np.nan).mean(axis=0)
    pair_stats = pd.DataFrame({
        "pair_id": pos.columns,
        "trades": trade_counts.values,
        "avg_hold_days": avg_hold.values
    })

    # 輸出單 combo 曲線
    ensure_dir(out_dir)
    eq_df = pd.DataFrame({"date": equity.index, "equity": equity.values, "ret": ret.values})
    eq_df["drawdown"] = dd.values
    eq_df.to_csv(os.path.join(out_dir, "equity_curve.csv"), index=False)
    pair_stats.to_csv(os.path.join(out_dir, "pair_stats.csv"), index=False)

    return ret, equity, m, pair_stats


# ====== 主流程 ======

def main():
    ap = argparse.ArgumentParser(description="Full baseline backtest across (formation_length × trading_period) combos (weekly-cache compatible).")
    # 路徑與集合
    ap.add_argument("--top-csv", type=str, default="cache/top_pairs.csv", help="Selection CSV with trading_period info.")
    ap.add_argument("--cache-root", type=str, default="cache/rolling_cache_v1", help="Rolling cache root (weekly: cache/rolling_cache_weekly_v1).")
    ap.add_argument("--price-type", type=str, default="log", choices=["raw","log"], help="Price type to use (weekly cache default: log).")
    ap.add_argument("--formation-lengths", type=str, default="all", help="Comma floats or 'all'.")
    ap.add_argument("--trading-periods", type=str, default="all", help="Comma period labels or 'all'.")
    ap.add_argument("--z-window", type=int, default=63, help="Z window; for weekly cache this is in weeks.")

    # 規則與成本
    ap.add_argument("--z-entry", type=float, default=2.0, help="Base entry threshold.")
    ap.add_argument("--z-exit", type=float, default=0.5, help="Base exit threshold.")
    ap.add_argument("--time-stop", type=int, default=40, help="Base time stop in trading days (overridden by --time-stop-weeks).")
    ap.add_argument("--time-stop-weeks", type=int, default=None, help="Time stop in weeks; converted to days as ceil(weeks*5) and overrides --time-stop.")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="Per-leg, one-way cost in bps.")

    # Top-K 與 gate
    ap.add_argument("--n-pairs-cap", type=int, default=60, help="Per-combo max Top-K if gate not provided.")
    ap.add_argument("--use-gate", action="store_true", help="Use noise-gate JSON for parameters.")
    ap.add_argument("--gate-json", type=str, default="reports/sanity/period_noise_gate.json", help="Gate JSON path.")
    ap.add_argument("--exclude-failed-gate", action="store_true", help="Skip combos with fail_period=True in gate JSON.")

    # 疊加模式
    ap.add_argument("--stack-mode", type=str, default="stack", choices=["stack","pick_best"], help="How to combine combos.")
    ap.add_argument("--pick-best-criterion", type=str, default="capital_scale", choices=["capital_scale","pairs_count"], help="Criterion when pick_best.")
    ap.add_argument("--drop-missing-threshold", type=float, default=0.5, help="Skip combo if missing cache ratio > threshold.")

    # 其他
    ap.add_argument("--capital", type=float, default=1_000_000.0, help="Total capital (for cost scaling; returns are scaled).")
    ap.add_argument("--out-dir", type=str, default="reports/baseline_full", help="Output directory.")
    ap.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Logging level.")

    # 週頻支援：是否忽略 selection 檔中的 formation_length 篩選（可用於讓同一份名單覆蓋多個 L）
    ap.add_argument("--ignore-selection-formation", action="store_true",
                    help="If set, ignore 'formation_length' in selection CSV when building combos (use same pairs across L).")

    args = ap.parse_args()
    setup_logger(args.log_level)
    ensure_dir(args.out_dir)

    # 週數 → 天數（覆蓋 time_stop）
    base_time_stop_days = int(args.time_stop)
    if args.time_stop_weeks is not None:
        base_time_stop_days = int(ceil(float(args.time_stop_weeks) * 5.0))
        logging.info(f"Using time_stop_weeks={args.time_stop_weeks} -> time_stop_days={base_time_stop_days}")

    # 載入選股
    if not os.path.isfile(args.top_csv):
        logging.error(f"top-csv not found: {args.top_csv}")
        sys.exit(2)
    sel = pd.read_csv(args.top_csv)
    sel = to_pair_id(sel)

    # 集合解析
    if args.formation_lengths.lower() == "all":
        if args.ignore_selection_formation:
            # 若忽略 selection 內的 formation_length，仍需一份 L 清單；此時從引數無法推得，請明確給值
            logging.error("When --ignore-selection-formation is set, please specify --formation-lengths explicitly.")
            sys.exit(2)
        L_list = sorted(sel["formation_length"].dropna().unique().astype(float).tolist())
    else:
        L_list = [float(x.strip()) for x in args.formation_lengths.split(",") if x.strip()]

    if args.trading_periods.lower() == "all":
        TP_list = sorted(sel["trading_period"].dropna().unique().astype(str).tolist())
    else:
        TP_list = [x.strip() for x in args.trading_periods.split(",") if x.strip()]

    logging.info(f"Combos target: L={L_list} TP={TP_list} z_window(weeks)={args.z_window}")

    # gate 載入
    gate_map = load_noise_gate_json(args.gate_json) if args.use_gate else {}

    # 依 combo 產生名單
    combo_rows = []
    for L in L_list:
        if args.ignore_selection_formation:
            sdf = sel[sel["trading_period"].isin(TP_list)].copy()
        else:
            sdf = sel[(sel["formation_length"] == L) & (sel["trading_period"].isin(TP_list))].copy()

        if sdf.empty:
            logging.warning(f"No pairs for L={L} after selection filtering.")
            continue

        for tp, g in sdf.groupby("trading_period"):
            # 自該 combo 的交易窗
            t_start = str(g["trading_start"].iloc[0]) if "trading_start" in g.columns else None
            t_end = str(g["trading_end"].iloc[0]) if "trading_end" in g.columns else None
            if not t_start or not t_end:
                logging.warning(f"Missing trading window for L={L} tp={tp}. Skipped.")
                continue

            # Gate 與 Top-K
            gkey = (float(L), str(tp))
            gate = gate_map.get(gkey, {}) if gate_map else {}
            if args.exclude_failed_gate and gate and bool(gate.get("fail_period", False)):
                logging.info(f"Combo L={L} tp={tp} excluded by gate (fail_period=True).")
                continue

            rec_topk = int(gate.get("rec_topk")) if gate and pd.notna(gate.get("rec_topk", np.nan)) else None
            use_topk = rec_topk if rec_topk is not None else int(args.n_pairs_cap)

            # 取 Top-K（rank_final 升冪）
            g_sorted = g.sort_values(["rank_final"], ascending=True)
            g_used = g_sorted.head(int(use_topk)).copy()
            pair_ids = g_used["pair_id"].unique().tolist()

            combo_rows.append(dict(
                L=float(L), tp=str(tp),
                trading_start=str(pd.to_datetime(t_start).date()),
                trading_end=str(pd.to_datetime(t_end).date()),
                gate=gate,
                pair_ids=pair_ids,
                used_topk=int(len(pair_ids)),
                capital_scale=float(gate.get("capital_scale", 1.0)) if gate else 1.0
            ))

    if not combo_rows:
        logging.error("No combos to run after filtering.")
        sys.exit(2)

    combos = pd.DataFrame(combo_rows)

    # pick_best 模式：每個 trading_period 僅保留一個 L
    if args.stack_mode == "pick_best":
        kept = []
        for tp, grp in combos.groupby("tp"):
            if args.pick_best_criterion == "capital_scale":
                # 以 gate 的 capital_scale 高者為優先；若無 gate，則 pairs_count 多者
                grp = grp.sort_values(["capital_scale", "used_topk"], ascending=[False, False])
            else:
                grp = grp.sort_values(["used_topk"], ascending=False)
            kept.append(grp.iloc[0])
        combos = pd.DataFrame(kept).reset_index(drop=True)
        logging.info(f"pick_best mode: kept combos={len(combos)} (one per trading_period)")

    # 建立 loader（每個 L 會重建一次，避免反覆更換 formation_length）
    results = []
    missing_records = []

    for (L, ), gL in combos.groupby(["L"]):
        loader = RollingCacheLoader(
            root=args.cache_root,
            price_type=args.price_type,
            formation_length=float(L),
            z_window=int(args.z_window),
            log_level=args.log_level
        )
        # 逐 combo 執行
        for _, row in gL.iterrows():
            tp = row["tp"]
            pair_ids = list(row["pair_ids"])
            # 檢查快取缺漏
            missing = loader.check_missing(pair_ids)
            miss_ratio = len(missing) / max(1, len(pair_ids))
            if miss_ratio > float(args.drop_missing_threshold):
                logging.warning(f"[Skip] L={L} tp={tp} missing_ratio={miss_ratio:.2%} > threshold.")
                missing_records.append(dict(L=L, tp=tp, missing=len(missing), total=len(pair_ids), ratio=miss_ratio))
                continue
            if missing:
                logging.info(f"[Warn] L={L} tp={tp} missing cache files: {len(missing)} (will drop).")
                pair_ids = [p for p in pair_ids if p not in missing]

            combo_tag = f"L={L}|tp={tp}"
            logging.info(f"[Combo] {combo_tag} pairs={len(pair_ids)} window={row['trading_start']}~{row['trading_end']}")

            # 單 combo 回測
            out_dir = os.path.join(args.out_dir, "combos", f"L{int(round(L*100)):03d}_{tp}")
            ret_c, eq_c, m_c, pair_stats = run_combo_backtest(
                pair_ids=pair_ids,
                trading_start=row["trading_start"],
                trading_end=row["trading_end"],
                loader=loader,
                base_z_entry=float(args.z_entry),
                base_z_exit=float(args.z_exit),
                base_time_stop=int(base_time_stop_days),
                gate_cfg=row["gate"] if args.use_gate else None,
                cost_bps=float(args.cost_bps),
                capital=float(args.capital),
                out_dir=out_dir,
                combo_tag=combo_tag
            )
            if len(ret_c) == 0:
                continue

            # 保存 combo 指標
            m_c.update(dict(
                formation_length=float(L),
                trading_period=str(tp),
                trading_start=str(row["trading_start"]),
                trading_end=str(row["trading_end"]),
                capital_scale=float(row["capital_scale"]),
                used_topk=int(row["used_topk"]),
                z_window_weeks=int(args.z_window)
            ))
            results.append(dict(ret=ret_c, eq=eq_c, metrics=m_c))

    if not results:
        logging.error("No combo results produced. Exit.")
        # 輸出缺漏表以利診斷
        if missing_records:
            pd.DataFrame(missing_records).to_csv(os.path.join(args.out_dir, "missing_cache_pairs.csv"), index=False)
        sys.exit(2)

    # 疊加：stack（活躍 combo 當日按 capital_scale 權重正規化）；pick_best（只有一個 L/期）
    # 先對齊所有日期
    all_dates = pd.Index([])
    for r in results:
        all_dates = all_dates.union(r["ret"].index)
    all_dates = all_dates.sort_values()

    # 準備 DataFrame
    mat = pd.DataFrame(index=all_dates)
    weights = pd.DataFrame(index=all_dates)
    combo_tags = []
    metrics_rows = []

    for r in results:
        L = r["metrics"]["formation_length"]
        tp = r["metrics"]["trading_period"]
        tag = f"L{int(round(L*100)):03d}_{tp}"
        combo_tags.append(tag)
        ret = r["ret"].reindex(all_dates).fillna(0.0)  # 非交易窗視為 0 報酬
        mat[tag] = ret.values
        # 活躍權重：在其交易窗內使用 capital_scale，窗外為 0
        start = pd.to_datetime(r["metrics"]["trading_start"])
        end = pd.to_datetime(r["metrics"]["trading_end"])
        w = pd.Series(0.0, index=all_dates)
        w[(all_dates >= start) & (all_dates <= end)] = float(r["metrics"]["capital_scale"])
        weights[tag] = w.values
        # 記錄指標
        metrics_rows.append(r["metrics"])

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["trading_period","formation_length"]).reset_index(drop=True)
    metrics_df.to_csv(os.path.join(args.out_dir, "combos_metrics.csv"), index=False)

    # 權重正規化（當日活躍 combo）
    w_active_sum = weights.sum(axis=1).replace(0.0, np.nan)
    weights_norm = weights.div(w_active_sum, axis=0).fillna(0.0)
    # 總投組日報酬
    ret_total = (mat * weights_norm).sum(axis=1)
    equity_total = (1.0 + ret_total).cumprod()
    dd_total = max_drawdown_curve(equity_total)
    m_total = ann_metrics(ret_total)
    m_total["max_drawdown"] = float(dd_total.min()) if len(dd_total) else 0.0
    # 每日活躍（權重 > 0）的 combo 數之平均
    m_total["avg_active_combos"] = float((weights > 0).sum(axis=1).mean())
    m_total["start"] = str(all_dates.min().date()) if len(all_dates) else ""
    m_total["end"] = str(all_dates.max().date()) if len(all_dates) else ""
    m_total["price_type"] = args.price_type
    m_total["z_window_weeks"] = int(args.z_window)
    m_total["stack_mode"] = args.stack_mode
    if args.time_stop_weeks is not None:
        m_total["time_stop_weeks"] = int(args.time_stop_weeks)

    # 輸出總曲線與 meta
    eq_total = pd.DataFrame({"date": all_dates, "equity": equity_total.values, "ret": ret_total.values})
    eq_total["drawdown"] = dd_total.values
    eq_total.to_csv(os.path.join(args.out_dir, "equity_curve_total.csv"), index=False)

    with open(os.path.join(args.out_dir, "total_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m_total, f, indent=2)

    if missing_records:
        pd.DataFrame(missing_records).to_csv(os.path.join(args.out_dir, "missing_cache_pairs.csv"), index=False)

    # run_meta
    run_meta = dict(
        formation_lengths=L_list,
        trading_periods=TP_list,
        use_gate=bool(args.use_gate),
        gate_json=str(args.gate_json),
        stack_mode=str(args.stack_mode),
        pick_best_criterion=str(args.pick_best_criterion),
        n_pairs_cap=int(args.n_pairs_cap),
        z_entry=float(args.z_entry),
        z_exit=float(args.z_exit),
        time_stop_days=int(base_time_stop_days),
        cost_bps=float(args.cost_bps),
        price_type=str(args.price_type),
        z_window_weeks=int(args.z_window),
        capital=float(args.capital),
        ignore_selection_formation=bool(args.ignore_selection_formation)
    )
    with open(os.path.join(args.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    logging.info(f"[WRITE] {os.path.join(args.out_dir, 'equity_curve_total.csv')}")
    logging.info(f"[WRITE] {os.path.join(args.out_dir, 'combos_metrics.csv')}")
    logging.info(f"[WRITE] {os.path.join(args.out_dir, 'total_metrics.json')}")
    logging.info(f"[WRITE] {os.path.join(args.out_dir, 'run_meta.json')}")
    logging.info(f"Backtest full done. Total Sharpe={m_total['sharpe']:.2f} AnnRet={m_total['ann_return']:.2%} "
                 f"AnnVol={m_total['ann_vol']:.2%} MDD={m_total['max_drawdown']:.2%}")

if __name__ == "__main__":
    main()