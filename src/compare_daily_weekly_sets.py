#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_daily_weekly_sets.py
- 讀取「日頻」與「週頻」的 sets summary（CSV），在同成本假設下，依 formation_length 與
  「日↔週等效 z-window 映射」成對比較。
- 僅用 summary 近似比較（未使用時間序列），提供：
  1) 風險對齊（target volatility，預設 10%）的年化報酬與 MDD 近似
  2) Sharpe 直接比較（已是風險標準化）
  3) 產出對照表 matched_comparison.csv
  4) 成對長條圖：
     - bars_sharpe_daily_vs_weekly.png
     - bars_ret_targetVol_daily_vs_weekly.png（target-vol > 0）
     - bars_mdd_targetVol_daily_vs_weekly.png（target-vol > 0，值愈小愈佳）
  5) 散佈圖：
     - scatter_sharpe_daily_vs_weekly.png
     - scatter_ret_targetVol_daily_vs_weekly.png（target-vol > 0）
  6) 勝率熱圖（按 formation_length 與按 family 匯總）：
     - heatmap_winrate_by_L.png、winrate_by_L.csv
     - heatmap_winrate_by_family.png、winrate_by_family.csv
  7) 漂亮對齊的成對列印與存檔：
     - paired_table_pretty.txt（對齊文字版）
     - paired_table.csv（精簡欄位，便於複製到論文表格）
  8) summary.txt（勝出次數與平均差）
- 注意：此為 summary 近似版（ret@TV 與 MDD@TV 以年化數字線性縮放近似），正式檢定需用時間序列重算。

使用者情境
- 週頻無 gate → 日頻比較時建議一律使用 scenario_id 末段 gateN。
- 主結果：成本 5（真實情境）；附錄：成本 0（理論上限）。

程式輸出（print/log）：英文；註解：繁體中文。
"""

import os
import sys
import argparse
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ====== 基本工具 ======

def setup_logger(level: str = "INFO"):
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

def parse_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1","true","t","yes","y"):
        return True
    if s in ("0","false","f","no","n",""):
        return False
    return False

def parse_mapping(s: str) -> Dict[int, int]:
    """
    解析日→週 z-window 映射字串，例如 "21:4,42:8,63:13,126:26,252:52"
    回傳 {daily_z: weekly_z}
    """
    out: Dict[int,int] = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        a, b = tok.split(":")
        out[int(float(a.strip()))] = int(float(b.strip()))
    return out

def setup_style(dpi: int = 150):
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9

def safe_div(a: float, b: float) -> float:
    """安全除法（b=0 時回傳 NaN）。"""
    try:
        b = float(b)
        if abs(b) <= 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

def fmt_pct(x: float, d: int = 2, dash: str = "—") -> str:
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return dash
        return f"{float(x)*100:.{d}f}%"
    except Exception:
        return dash

def fmt_dec(x: float, d: int = 2, dash: str = "—") -> str:
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return dash
        return f"{float(x):.{d}f}"
    except Exception:
        return dash


# ====== 主流程 ======

def main():
    ap = argparse.ArgumentParser(description="Compare daily vs weekly sets summary with window mapping and risk-aligned metrics.")
    ap.add_argument("--daily-csv", type=str, required=True, help="Path to daily_sets_summary.csv")
    ap.add_argument("--weekly-csv", type=str, required=True, help="Path to weekly_sets_summary.csv")

    # Daily 過濾條件（預設以 scenario_id 準確匹配；若未提供則以欄位 fallback 過濾）
    ap.add_argument("--daily-scenario-id", type=str, default=None, help="Filter daily by scenario_id (e.g., price_log__cost5__gateN)")
    ap.add_argument("--daily-price-type", type=str, default="log", help="Fallback filter: price_type for daily (if no scenario_id)")
    ap.add_argument("--daily-cost-bps", type=float, default=5.0, help="Fallback filter: cost_bps for daily")
    ap.add_argument("--daily-use-gate", type=str, default="False", help="Fallback filter: use_gate for daily (True/False). Weekly has no gate, so default False.")

    # Weekly 過濾條件
    ap.add_argument("--weekly-cost-bps", type=float, default=None, help="Filter weekly by cost_bps (default = daily-cost-bps)")

    # 風險對齊（目標年化波動）
    ap.add_argument("--target-vol", type=float, default=0.10, help="Target annualized volatility (0.10=10%%). 0 disables ret@TV & MDD@TV plots.")

    # 視窗映射（日→週）
    ap.add_argument("--mapping", type=str, default="21:4,42:8,63:13,126:26,252:52", help="Daily-to-weekly z-window mapping")

    # 重複鍵取捨
    ap.add_argument("--select-rule", type=str, default="sharpe", choices=["sharpe","ann_return"], help="If duplicates per (L,z) exist, pick the row with highest metric.")

    # 輸出與圖表
    ap.add_argument("--out-dir", type=str, default="reports/summary/compare_daily_weekly", help="Output directory")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")

    args = ap.parse_args()

    setup_logger("INFO")
    setup_style(args.dpi)
    ensure_dir(args.out_dir)

    # ===== 載入資料 =====
    if not os.path.isfile(args.daily_csv):
        print(f"[ERROR] daily-csv not found: {args.daily_csv}")
        sys.exit(2)
    if not os.path.isfile(args.weekly_csv):
        print(f"[ERROR] weekly-csv not found: {args.weekly_csv}")
        sys.exit(2)

    daily = pd.read_csv(args.daily_csv)
    weekly = pd.read_csv(args.weekly_csv)

    # 轉型
    if "use_gate" in daily.columns:
        daily["use_gate"] = daily["use_gate"].apply(parse_bool)

    # ===== Daily 過濾 =====
    if args.daily_scenario_id:
        dflt = daily[daily["scenario_id"].astype(str) == str(args.daily_scenario_id)].copy()
        print(f"[INFO] Daily filter by scenario_id={args.daily_scenario_id}: rows={len(dflt)}")
    else:
        ug = parse_bool(args.daily_use_gate)
        dflt = daily[
            (daily.get("price_type","").astype(str) == str(args.daily_price_type)) &
            (daily["cost_bps"].astype(float) == float(args.daily_cost_bps)) &
            (daily["use_gate"].astype(bool) == bool(ug))
        ].copy()
        print(f"[INFO] Daily filter by price_type={args.daily_price_type}, cost_bps={args.daily_cost_bps}, use_gate={ug}: rows={len(dflt)}")

    if dflt.empty:
        print("[ERROR] No daily rows after filtering.")
        sys.exit(2)

    # ===== Weekly 過濾 =====
    wc = float(args.daily_cost_bps) if args.weekly_cost_bps is None else float(args.weekly_cost_bps)
    wflt = weekly[(weekly["cost_bps"].astype(float) == float(wc))].copy()
    print(f"[INFO] Weekly filter by cost_bps={wc}: rows={len(wflt)}")
    if wflt.empty:
        print("[ERROR] No weekly rows after filtering.")
        sys.exit(2)

    # ===== 去重（同鍵多列時，挑表現最好者） =====
    key_daily = ["formation_length","z_window"]
    key_weekly = ["formation_length","z_window"]

    def pick_best(df: pd.DataFrame, by: str, key_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        if by not in df.columns:
            return df.drop_duplicates(subset=key_cols, keep="first")
        df = df.sort_values(by=by, ascending=False)
        return df.drop_duplicates(subset=key_cols, keep="first")

    dsel = pick_best(dflt, by=str(args.select_rule), key_cols=key_daily)
    wsel = pick_best(wflt, by=str(args.select_rule), key_cols=key_weekly)

    # ===== 建立日→週映射，並對齊鍵 =====
    mapping = parse_mapping(args.mapping)  # e.g., {21:4, 42:8, 63:13, 126:26, 252:52}
    if len(mapping) == 0:
        print("[ERROR] Invalid mapping.")
        sys.exit(2)

    dsel["weekly_z_window"] = dsel["z_window"].map(mapping)
    dsel = dsel.dropna(subset=["weekly_z_window"])
    dsel["weekly_z_window"] = dsel["weekly_z_window"].astype(int)

    # 左表（日）欄位重新命名
    left = dsel.rename(columns={
        "formation_length":"L",
        "z_window":"daily_z",
        "ann_return":"daily_ann_return",
        "ann_vol":"daily_ann_vol",
        "sharpe":"daily_sharpe",
        "max_drawdown":"daily_mdd",
        "cum_return":"daily_cum_return",
        "win_rate":"daily_win_rate",
        "profit_factor":"daily_pf",
        "total_trades":"daily_trades",
        "avg_duration_days":"daily_avg_dur"
    })
    # 右表（週）欄位重新命名
    right = wsel.rename(columns={
        "formation_length":"L",
        "z_window":"weekly_z",
        "ann_return":"weekly_ann_return",
        "ann_vol":"weekly_ann_vol",
        "sharpe":"weekly_sharpe",
        "max_drawdown":"weekly_mdd",
        "cum_return":"weekly_cum_return",
        "win_rate":"weekly_win_rate",
        "profit_factor":"weekly_pf",
        "total_trades":"weekly_trades",
        "avg_duration_days":"weekly_avg_dur"
    })

    # 合併（以 L 與對映後的週 z_window 對齊）
    m = pd.merge(
        left,
        right[["L","weekly_z","weekly_ann_return","weekly_ann_vol","weekly_sharpe","weekly_mdd",
               "weekly_cum_return","weekly_win_rate","weekly_pf","weekly_trades","weekly_avg_dur"]],
        how="inner",
        left_on=["L","weekly_z_window"],
        right_on=["L","weekly_z"]
    )

    if m.empty:
        print("[ERROR] No matched rows after join. Check mapping or filters.")
        sys.exit(2)

    # 家族字串（好看用）
    m["family"] = m.apply(lambda r: f"{int(r['daily_z'])}↔{int(r['weekly_z'])}", axis=1)

    # ===== 風險對齊（目標波動） =====
    target = float(args.target_vol)
    if target > 0:
        m["daily_ret_at_tv"]  = m.apply(lambda r: r["daily_ann_return"]  * safe_div(target, r["daily_ann_vol"]), axis=1)
        m["weekly_ret_at_tv"] = m.apply(lambda r: r["weekly_ann_return"] * safe_div(target, r["weekly_ann_vol"]), axis=1)
        m["daily_mdd_abs"]    = m["daily_mdd"].abs()
        m["weekly_mdd_abs"]   = m["weekly_mdd"].abs()
        m["daily_mdd_at_tv"]  = m.apply(lambda r: r["daily_mdd_abs"]  * safe_div(target, r["daily_ann_vol"]), axis=1)
        m["weekly_mdd_at_tv"] = m.apply(lambda r: r["weekly_mdd_abs"] * safe_div(target, r["weekly_ann_vol"]), axis=1)
    else:
        m["daily_ret_at_tv"] = np.nan
        m["weekly_ret_at_tv"] = np.nan
        m["daily_mdd_at_tv"] = np.nan
        m["weekly_mdd_at_tv"] = np.nan
        m["daily_mdd_abs"] = m["daily_mdd"].abs()
        m["weekly_mdd_abs"] = m["weekly_mdd"].abs()

    # ===== 差異欄位（Weekly − Daily；MDD 以較小為佳，故用 weekly - daily 但解讀要注意） =====
    m["delta_sharpe"] = m["weekly_sharpe"] - m["daily_sharpe"]
    m["delta_ret_at_tv"] = m["weekly_ret_at_tv"] - m["daily_ret_at_tv"]
    m["delta_mdd_at_tv"] = m["weekly_mdd_at_tv"] - m["daily_mdd_at_tv"]  # 負值代表週頻回撤較淺（較好）

    # 排序
    m = m.sort_values(["L","daily_z"]).reset_index(drop=True)

    # ===== 匯出 matched_comparison.csv =====
    cols = [
        "L","family","daily_z","weekly_z",
        "daily_ann_return","daily_ann_vol","daily_sharpe","daily_mdd","daily_mdd_abs",
        "weekly_ann_return","weekly_ann_vol","weekly_sharpe","weekly_mdd","weekly_mdd_abs",
        "daily_ret_at_tv","weekly_ret_at_tv","delta_ret_at_tv",
        "daily_mdd_at_tv","weekly_mdd_at_tv","delta_mdd_at_tv",
        "delta_sharpe",
        "daily_cum_return","weekly_cum_return",
        "daily_trades","weekly_trades","daily_win_rate","weekly_win_rate","daily_pf","weekly_pf",
        "daily_avg_dur","weekly_avg_dur"
    ]
    out_csv = os.path.join(args.out_dir, "matched_comparison.csv")
    m[cols].to_csv(out_csv, index=False)
    print(f"[WRITE] {out_csv}")

    # ===== 精簡版成對表（CSV）＋ 漂亮對齊列印（TXT + console） =====
    paired_cols = [
        "L","family","daily_z","weekly_z",
        "daily_sharpe","weekly_sharpe","delta_sharpe",
        "daily_ret_at_tv","weekly_ret_at_tv","delta_ret_at_tv",
        "daily_mdd_at_tv","weekly_mdd_at_tv","delta_mdd_at_tv"
    ]
    paired_df = m[paired_cols].copy()
    paired_df.to_csv(os.path.join(args.out_dir, "paired_table.csv"), index=False)
    print(f"[WRITE] {os.path.join(args.out_dir, 'paired_table.csv')}")

    def print_pretty_table(df: pd.DataFrame, target: float, out_txt: str):
        # 欄寬設定
        w = {
            "L": 5, "family": 8,
            "dz": 5, "wz": 5,
            "ret": 10, "sh": 9, "mdd": 10,
            "dret": 10, "dsh": 9, "dmdd": 10
        }
        title = f"=== Daily vs. Weekly Sets (matched L & family; cost_bps={wc:g}, target_vol={'NA' if target<=0 else f'{int(round(target*100))}%'} ) ==="
        lines = [title]
        header = f"{'L':>{w['L']}}  {'family':<{w['family']}}  {'D_z':>{w['dz']}}  {'D_ret@TV':>{w['ret']}}  {'D_Sharpe':>{w['sh']}}  {'D_MDD@TV':>{w['mdd']}}   {'W_z':>{w['wz']}}  {'W_ret@TV':>{w['ret']}}  {'W_Sharpe':>{w['sh']}}  {'W_MDD@TV':>{w['mdd']}}   {'Δret@TV':>{w['dret']}}  {'ΔSharpe':>{w['dsh']}}  {'ΔMDD@TV':>{w['dmdd']}}"
        lines.append(header)
        lines.append("-" * len(header))
        for _, r in df.iterrows():
            line = (
                f"{float(r['L']):>{w['L']}.1f}  "
                f"{str(r['family']):<{w['family']}}  "
                f"{int(r['daily_z']):>{w['dz']}}  "
                f"{fmt_pct(r['daily_ret_at_tv']):>{w['ret']}}  "
                f"{fmt_dec(r['daily_sharpe']):>{w['sh']}}  "
                f"{fmt_pct(r['daily_mdd_at_tv']):>{w['mdd']}}   "
                f"{int(r['weekly_z']):>{w['wz']}}  "
                f"{fmt_pct(r['weekly_ret_at_tv']):>{w['ret']}}  "
                f"{fmt_dec(r['weekly_sharpe']):>{w['sh']}}  "
                f"{fmt_pct(r['weekly_mdd_at_tv']):>{w['mdd']}}   "
                f"{fmt_dec(r['delta_ret_at_tv']):>{w['dret']}}  "
                f"{fmt_dec(r['delta_sharpe']):>{w['dsh']}}  "
                f"{fmt_dec(r['delta_mdd_at_tv']):>{w['dmdd']}}"
            )
            lines.append(line)
        txt = "\n".join(lines)
        print(txt)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"[WRITE] {out_txt}")

    print_pretty_table(paired_df, target=target, out_txt=os.path.join(args.out_dir, "paired_table_pretty.txt"))

    # ===== 整體摘要（summary.txt） =====
    def _wins(x: pd.Series, greater_is_better: bool = True):
        x = pd.to_numeric(x, errors="coerce")
        if greater_is_better:
            return int((x > 0).sum()), int((x < 0).sum()), int((x == 0).sum())
        else:
            return int((x < 0).sum()), int((x > 0).sum()), int((x == 0).sum())

    lines = []
    lines.append(f"Matched pairs: {len(m)}")
    # Sharpe
    sh_win, sh_lose, sh_tie = _wins(m["delta_sharpe"], greater_is_better=True)
    lines.append(f"Sharpe: weekly-daily avg delta={pd.to_numeric(m['delta_sharpe']).mean():.3f} | wins={sh_win}, loses={sh_lose}, ties={sh_tie}")
    # ret@TV
    if target > 0 and m["delta_ret_at_tv"].notna().any():
        rt_win, rt_lose, rt_tie = _wins(m["delta_ret_at_tv"], greater_is_better=True)
        lines.append(f"Return@{int(round(target*100))}% vol: weekly-daily avg delta={pd.to_numeric(m['delta_ret_at_tv']).mean():.3f} | wins={rt_win}, loses={rt_lose}, ties={rt_tie}")
    else:
        lines.append("Return@TV: disabled or NA (target-vol=0)")
    # MDD@TV（小為佳 → greater_is_better=False）
    if target > 0 and m["delta_mdd_at_tv"].notna().any():
        md_win, md_lose, md_tie = _wins(m["delta_mdd_at_tv"], greater_is_better=False)
        lines.append(f"MDD@{int(round(target*100))}% vol (smaller is better): weekly-daily avg delta={pd.to_numeric(m['delta_mdd_at_tv']).mean():.3f} | wins={md_win}, loses={md_lose}, ties={md_tie}")
    else:
        lines.append("MDD@TV: disabled or NA (target-vol=0)")

    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[WRITE] " + os.path.join(args.out_dir, "summary.txt"))
    for ln in lines:
        print("[SUMMARY] " + ln)

    # ===== 視覺化：標籤 =====
    labels = [f"L{float(L):.1f}-D{int(dz)}→W{int(wz)}" for L, dz, wz in zip(m["L"], m["daily_z"], m["weekly_z"])]
    x = np.arange(len(labels))

    # ===== 成對長條：Sharpe =====
    try:
        fig, ax = plt.subplots(figsize=(max(7, 0.45*len(labels) + 2), 4))
        w = 0.38
        r1 = ax.bar(x - w/2, m["daily_sharpe"].values, width=w, label="Daily")
        r2 = ax.bar(x + w/2, m["weekly_sharpe"].values, width=w, label="Weekly")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Sharpe: Daily vs Weekly (matched L & windows)")
        ax.set_ylabel("Sharpe")
        ax.legend()
        plt.tight_layout()
        fp = os.path.join(args.out_dir, "bars_sharpe_daily_vs_weekly.png")
        plt.savefig(fp); plt.close()
        print(f"[WRITE] {fp}")
    except Exception as e:
        logging.warning(f"bars_sharpe plot failed: {repr(e)}")

    # ===== 成對長條：ret@targetVol（target>0） =====
    if target > 0:
        try:
            dfp = m.dropna(subset=["daily_ret_at_tv","weekly_ret_at_tv"]).copy()
            labels_tv = [f"L{float(L):.1f}-D{int(dz)}→W{int(wz)}" for L, dz, wz in zip(dfp["L"], dfp["daily_z"], dfp["weekly_z"])]
            x2 = np.arange(len(labels_tv))
            fig, ax = plt.subplots(figsize=(max(7, 0.45*len(labels_tv) + 2), 4))
            w = 0.38
            r1 = ax.bar(x2 - w/2, (dfp["daily_ret_at_tv"]*100.0).values, width=w, label=f"Daily @{int(round(target*100))}% vol")
            r2 = ax.bar(x2 + w/2, (dfp["weekly_ret_at_tv"]*100.0).values, width=w, label=f"Weekly @{int(round(target*100))}% vol")
            ax.set_xticks(x2)
            ax.set_xticklabels(labels_tv, rotation=45, ha="right")
            ax.set_title(f"Return at {int(round(target*100))}% Vol: Daily vs Weekly")
            ax.set_ylabel("Return (%)")
            ax.legend()
            plt.tight_layout()
            fp = os.path.join(args.out_dir, "bars_ret_targetVol_daily_vs_weekly.png")
            plt.savefig(fp); plt.close()
            print(f"[WRITE] {fp}")
        except Exception as e:
            logging.warning(f"bars_ret_targetVol plot failed: {repr(e)}")

    # ===== 成對長條：MDD@targetVol（target>0，小者為佳） =====
    if target > 0:
        try:
            dfp = m.dropna(subset=["daily_mdd_at_tv","weekly_mdd_at_tv"]).copy()
            labels_tv = [f"L{float(L):.1f}-D{int(dz)}→W{int(wz)}" for L, dz, wz in zip(dfp["L"], dfp["daily_z"], dfp["weekly_z"])]
            x3 = np.arange(len(labels_tv))
            fig, ax = plt.subplots(figsize=(max(7, 0.45*len(labels_tv) + 2), 4))
            w = 0.38
            y1 = (dfp["daily_mdd_at_tv"].abs()*100.0).values
            y2 = (dfp["weekly_mdd_at_tv"].abs()*100.0).values
            r1 = ax.bar(x3 - w/2, y1, width=w, label=f"Daily MDD @{int(round(target*100))}% vol")
            r2 = ax.bar(x3 + w/2, y2, width=w, label=f"Weekly MDD @{int(round(target*100))}% vol")
            ax.set_xticks(x3)
            ax.set_xticklabels(labels_tv, rotation=45, ha="right")
            ax.set_title(f"Max Drawdown at {int(round(target*100))}% Vol (smaller is better)")
            ax.set_ylabel("Drawdown (%)")
            ax.legend()
            plt.tight_layout()
            fp = os.path.join(args.out_dir, "bars_mdd_targetVol_daily_vs_weekly.png")
            plt.savefig(fp); plt.close()
            print(f"[WRITE] {fp}")
        except Exception as e:
            logging.warning(f"bars_mdd_targetVol plot failed: {repr(e)}")

    # ===== 散佈圖：Sharpe =====
    try:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(m["weekly_sharpe"], m["daily_sharpe"], alpha=0.8)
        lims = [
            min(np.nanmin(m["weekly_sharpe"]), np.nanmin(m["daily_sharpe"])),
            max(np.nanmax(m["weekly_sharpe"]), np.nanmax(m["daily_sharpe"]))
        ]
        if not np.isfinite(lims).all():
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.6)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Weekly Sharpe"); ax.set_ylabel("Daily Sharpe")
        ax.set_title("Sharpe Scatter (y=x reference)")
        plt.tight_layout()
        fp = os.path.join(args.out_dir, "scatter_sharpe_daily_vs_weekly.png")
        plt.savefig(fp); plt.close()
        print(f"[WRITE] {fp}")
    except Exception as e:
        logging.warning(f"scatter_sharpe plot failed: {repr(e)}")

    # ===== 散佈圖：ret@targetVol（target>0） =====
    if target > 0:
        try:
            dfp = m.dropna(subset=["daily_ret_at_tv","weekly_ret_at_tv"]).copy()
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.scatter(dfp["weekly_ret_at_tv"], dfp["daily_ret_at_tv"], alpha=0.8)
            lims = [
                min(np.nanmin(dfp["weekly_ret_at_tv"]), np.nanmin(dfp["daily_ret_at_tv"])),
                max(np.nanmax(dfp["weekly_ret_at_tv"]), np.nanmax(dfp["daily_ret_at_tv"]))
            ]
            if not np.isfinite(lims).all():
                lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, "k--", alpha=0.6)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel(f"Weekly Return @ {int(round(target*100))}% vol")
            ax.set_ylabel(f"Daily Return @ {int(round(target*100))}% vol")
            ax.set_title(f"Return @ {int(round(target*100))}% Vol (y=x reference)")
            plt.tight_layout()
            fp = os.path.join(args.out_dir, "scatter_ret_targetVol_daily_vs_weekly.png")
            plt.savefig(fp); plt.close()
            print(f"[WRITE] {fp}")
        except Exception as e:
            logging.warning(f"scatter_ret_targetVol plot failed: {repr(e)}")

    # ===== 勝率熱圖（按 formation_length 匯總） =====
    try:
        dfL = m.copy()
        grp = dfL.groupby("L")

        rows = []
        for L, g in grp:
            n_sh = int(g["delta_sharpe"].notna().sum())
            n_rt = int(g["delta_ret_at_tv"].notna().sum()) if target > 0 else 0
            n_md = int(g["delta_mdd_at_tv"].notna().sum()) if target > 0 else 0

            sh_win_rate = float((g["delta_sharpe"] > 0).mean()) * 100.0 if n_sh > 0 else np.nan
            rt_win_rate = float((g["delta_ret_at_tv"] > 0).mean()) * 100.0 if n_rt > 0 else np.nan
            md_win_rate = float((g["delta_mdd_at_tv"] < 0).mean()) * 100.0 if n_md > 0 else np.nan

            rows.append(dict(
                L=float(L),
                win_rate_ret_at_tv=rt_win_rate,
                win_rate_sharpe=sh_win_rate,
                win_rate_mdd_at_tv=md_win_rate
            ))

        wr = pd.DataFrame(rows).sort_values("L").set_index("L")
        out_wr_csv = os.path.join(args.out_dir, "winrate_by_L.csv")
        wr.to_csv(out_wr_csv)
        print(f"[WRITE] {out_wr_csv}")

        plt.figure(figsize=(6, max(3.5, 0.5*len(wr))))
        data = wr[["win_rate_ret_at_tv","win_rate_sharpe","win_rate_mdd_at_tv"]]
        sns.heatmap(
            data,
            annot=True, fmt=".0f",
            cmap="YlGnBu", cbar_kws={"label":"Win rate (%)"},
            vmin=0, vmax=100
        )
        plt.title("Weekly win rate by formation length")
        plt.xlabel("Metric")
        plt.ylabel("Formation length (years)")
        fp = os.path.join(args.out_dir, "heatmap_winrate_by_L.png")
        plt.tight_layout(); plt.savefig(fp); plt.close()
        print(f"[WRITE] {fp}")
    except Exception as e:
        logging.warning(f"winrate heatmap by L failed: {repr(e)}")

    # ===== 勝率熱圖（按 family 匯總） =====
    try:
        dff = m.copy()
        grp = dff.groupby("family")

        rows = []
        for fam, g in grp:
            n_sh = int(g["delta_sharpe"].notna().sum())
            n_rt = int(g["delta_ret_at_tv"].notna().sum()) if target > 0 else 0
            n_md = int(g["delta_mdd_at_tv"].notna().sum()) if target > 0 else 0

            sh_win_rate = float((g["delta_sharpe"] > 0).mean()) * 100.0 if n_sh > 0 else np.nan
            rt_win_rate = float((g["delta_ret_at_tv"] > 0).mean()) * 100.0 if n_rt > 0 else np.nan
            md_win_rate = float((g["delta_mdd_at_tv"] < 0).mean()) * 100.0 if n_md > 0 else np.nan

            rows.append(dict(
                family=str(fam),
                win_rate_ret_at_tv=rt_win_rate,
                win_rate_sharpe=sh_win_rate,
                win_rate_mdd_at_tv=md_win_rate
            ))

        wrf = pd.DataFrame(rows).sort_values("family").set_index("family")
        out_wrf_csv = os.path.join(args.out_dir, "winrate_by_family.csv")
        wrf.to_csv(out_wrf_csv)
        print(f"[WRITE] {out_wrf_csv}")

        plt.figure(figsize=(6, max(3.5, 0.5*len(wrf))))
        data = wrf[["win_rate_ret_at_tv","win_rate_sharpe","win_rate_mdd_at_tv"]]
        sns.heatmap(
            data,
            annot=True, fmt=".0f",
            cmap="YlGnBu", cbar_kws={"label":"Win rate (%)"},
            vmin=0, vmax=100
        )
        plt.title("Weekly win rate by family (D↔W window)")
        plt.xlabel("Metric")
        plt.ylabel("Family (D↔W)")
        fp = os.path.join(args.out_dir, "heatmap_winrate_by_family.png")
        plt.tight_layout(); plt.savefig(fp); plt.close()
        print(f"[WRITE] {fp}")
    except Exception as e:
        logging.warning(f"winrate heatmap by family failed: {repr(e)}")

    print("Done.")

if __name__ == "__main__":
    main()