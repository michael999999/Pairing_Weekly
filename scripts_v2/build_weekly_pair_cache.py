#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
以週收盤對數價進行滾動 OLS 與 z-score 計算，並將 α/β 對齊至日頻（從 T+1 起 forward-fill），
可一次建立多個 formation_length × z_window_weeks 的週頻 pair-cache。

變更與強化：
- 統一 ddof（cov/var/std 同一個 ddof，預設 1，可由 --ddof 指定）
- 新增 alpha_daily（週末估計，從 T+1 起 forward-fill）
- 新增 z_t1（週末 z shift 到下一交易日，便於直接用於 T+1 下單；且僅在 t1_tradeable_flag 為 True 時保留）
- t1_tradeable_flag 預設使用 weekly 的 T1_Close 判斷（可用參數切換為日頻判斷）
- meta/manifest 增補 ddof 與輸出欄位旗標（並標記 alpha_column = "alpha_daily"）
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import shutil
from datetime import datetime, timezone


# ====== 小工具：字串參數解析 ======

def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ====== pair 欄位處理 ======

def to_pair_id_df(df: pd.DataFrame) -> pd.DataFrame:
    """保證 pair_id 欄位存在；若只有 pair 字串，解析為 stock1/stock2。並對 stock1/stock2 做 strip。"""
    if "stock1" in df.columns and "stock2" in df.columns:
        df["stock1"] = df["stock1"].astype(str).str.strip()
        df["stock2"] = df["stock2"].astype(str).str.strip()
        df["pair_id"] = df["stock1"].astype(str) + "__" + df["stock2"].astype(str)
        return df
    if "pair" in df.columns:
        def _parse_pair(s: str) -> Tuple[str, str]:
            s = str(s).strip()
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            s = s.replace("'", "").replace('"', "")
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 2:
                return parts[0], parts[1]
            return "", ""
        tmp = df["pair"].apply(_parse_pair)
        df["stock1"] = tmp.apply(lambda t: str(t[0]).strip())
        df["stock2"] = tmp.apply(lambda t: str(t[1]).strip())
        df["pair_id"] = df["stock1"].astype(str) + "__" + df["stock2"].astype(str)
        return df
    raise ValueError("Input CSV must contain 'stock1' and 'stock2' or 'pair' column.")


# ====== 統計計算 ======

def rolling_ols_y_on_x(y: pd.Series, x: pd.Series, win: int, ddof: int = 1) -> Tuple[pd.Series, pd.Series]:
    """用週頻對數價執行滾動 OLS（y=alpha+beta*x），回傳 alpha, beta（只在滿足窗口時有值）。
    為避免 pandas 版本差異造成 ddof 不一致，這裡以 E[XY]-EX*EY 形式顯式計算 cov 與 var，並依 ddof 做縮放。
    - ddof=0：母體統計（cov = E[XY]-EX*EY；var = E[X^2]-(EX)^2）
    - ddof=1：樣本統計（上述值 × n/(n-1)；在 min_periods=win 前提下 n=win）
    """
    y = y.copy()
    x = x.reindex_like(y)

    # 週內均值
    mean_x = x.rolling(win, min_periods=win).mean()
    mean_y = y.rolling(win, min_periods=win).mean()

    # 以期望值形式計算 cov 與 var（先算母體，必要時再乘 n/(n-ddof)）
    ex = mean_x
    ey = mean_y
    exy = (x * y).rolling(win, min_periods=win).mean()
    ex2 = (x * x).rolling(win, min_periods=win).mean()

    cov0 = exy - (ex * ey)           # 母體 cov
    var0 = ex2 - (ex * ex)           # 母體 var

    if ddof == 1:
        # 在 min_periods=win 下，視窗大小 n 固定為 win
        factor = win / (win - 1) if win > 1 else np.nan
        cov_xy = cov0 * factor
        var_x = var0 * factor
    else:
        cov_xy = cov0
        var_x = var0

    # 避免 0 或負微小數造成除零/反常（理論上 var>=0，但浮點會有 -1e-16 類）
    var_x = var_x.where(var_x > 0.0, np.nan)

    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    return alpha, beta


def rolling_zscore(s: pd.Series, win: int, ddof: int = 1) -> pd.Series:
    """以窗口 win（週數）計算滾動 z-score；std 與 OLS 採用相同 ddof。"""
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std(ddof=ddof)
    return (s - mu) / sd.replace(0.0, np.nan)


# ====== 主要建置流程 ======

def build_one_combo(px_daily: pd.DataFrame,
                    px_daily_log: pd.DataFrame,
                    wk: pd.DataFrame,
                    pairs_df: pd.DataFrame,
                    formation_length: float,
                    z_window_weeks: int,
                    out_dir: Path,
                    overwrite_mode: str = "overwrite",
                    ddof: int = 1,
                    use_weekly_t1_for_tradeable: bool = True,
                    include_alpha_daily: bool = True,
                    include_z_t1: bool = True) -> Tuple[int, int]:
    """建立單一 L × Z 組合的所有 pair 快取，回傳 (built, skipped)。"""
    daily_index = px_daily.index

    # 週末索引與 W_Close/T1_Close 存取輔助
    if not isinstance(wk.columns, pd.MultiIndex):
        raise ValueError("weekly file must be a MultiIndex columns with Field including 'W_Close' and 'T1_Close'.")
    if "W_Close" not in wk.columns.get_level_values(1):
        raise ValueError("weekly file must include 'W_Close' field.")
    if "T1_Close" not in wk.columns.get_level_values(1):
        raise ValueError("weekly file must include 'T1_Close' field.")
    wk = wk.sort_index()
    week_end_index = wk.index

    # 清理模式
    if overwrite_mode == "clean" and out_dir.exists():
        print(f"[INFO] Cleaning output dir: {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    # 規模資訊
    win_weeks_form = int(math.ceil(float(formation_length) * 52.0))
    win_weeks_z = int(z_window_weeks)
    print(f"[INFO] Building combo L={formation_length}y (form={win_weeks_form}w), Z={win_weeks_z}w, ddof={ddof} -> {out_dir}")

    # 預計數
    built = 0
    skipped = 0

    # 快速映射：每週末日對應在日頻索引的位置
    wk_pos = daily_index.get_indexer(week_end_index)
    valid_wk_mask = wk_pos >= 0
    wk_pos_valid = wk_pos[valid_wk_mask]

    for _, row in pairs_df.iterrows():
        s1 = str(row["stock1"])
        s2 = str(row["stock2"])
        pid = str(row["pair_id"])

        # 目標檔案路徑
        out_path = pairs_dir / f"{pid}.pkl"
        if out_path.exists() and overwrite_mode == "skip":
            skipped += 1
            continue

        # 價格可用性（確認在日頻存在）
        if s1 not in px_daily.columns or s2 not in px_daily.columns:
            print(f"[WARN] Skip {pid}: symbol missing in daily prices.")
            skipped += 1
            continue
        try:
            x_w = wk[(s1, "W_Close")].copy()
            y_w = wk[(s2, "W_Close")].copy()
            x_t1_w = wk[(s1, "T1_Close")].copy()
            y_t1_w = wk[(s2, "T1_Close")].copy()
        except KeyError:
            print(f"[WARN] Skip {pid}: W_Close/T1_Close not found for one of symbols in weekly file.")
            skipped += 1
            continue

        # 以週頻對數價做 OLS 與殘差 z（窗口：週；ddof 統一）
        x_w_log = np.log(x_w.replace(0.0, np.nan))
        y_w_log = np.log(y_w.replace(0.0, np.nan))
        alpha_w, beta_w = rolling_ols_y_on_x(y_w_log, x_w_log, win_weeks_form, ddof=ddof)
        spread_w = y_w_log - (alpha_w + beta_w * x_w_log)
        z_w = rolling_zscore(spread_w, win_weeks_z, ddof=ddof)

        # 對齊到日頻 — 週末旗標
        week_end_flag = pd.Series(False, index=daily_index)
        week_end_flag.iloc[wk_pos_valid] = True

        # z：僅在週末日填值（values 對齊 valid mask）
        z_daily = pd.Series(np.nan, index=daily_index)
        z_vals = z_w.values  # 與 week_end_index 對齊
        z_daily.iloc[wk_pos_valid] = z_vals[valid_wk_mask]

        # 準備 T+1 索引（僅限能對到日頻的週末）
        pos_t1_all = wk_pos_valid + 1
        t1_in_range_mask = pos_t1_all < len(daily_index)
        pos_t1_valid = pos_t1_all[t1_in_range_mask]

        # β（T+1 起 forward-fill）：向量化寫法
        beta_daily = pd.Series(np.nan, index=daily_index)
        beta_vals = beta_w.values  # 與 week_end_index 對齊
        beta_vals_aligned = beta_vals[valid_wk_mask]
        beta_vals_t1 = beta_vals_aligned[t1_in_range_mask]
        beta_daily.iloc[pos_t1_valid] = beta_vals_t1
        beta_daily = beta_daily.ffill()

        # α（T+1 起 forward-fill）：與 β 相同規則（可選）
        alpha_daily = pd.Series(np.nan, index=daily_index)
        if include_alpha_daily:
            alpha_vals = alpha_w.values  # 與 week_end_index 對齊
            alpha_vals_aligned = alpha_vals[valid_wk_mask]
            alpha_vals_t1 = alpha_vals_aligned[t1_in_range_mask]
            alpha_daily.iloc[pos_t1_valid] = alpha_vals_t1
            alpha_daily = alpha_daily.ffill()

        # T+1 可交易旗標：預設使用 weekly 的 T1_Close（可選回退到日頻）
        t1_tradeable_flag = pd.Series(False, index=daily_index)
        if use_weekly_t1_for_tradeable:
            t1_valid_weekly = (x_t1_w.notna() & y_t1_w.notna()).values  # 與 week_end_index 對齊
            t1_valid_weekly_aligned = t1_valid_weekly[valid_wk_mask]
            t1_valid_weekly_t1 = t1_valid_weekly_aligned[t1_in_range_mask]
            t1_tradeable_flag.iloc[pos_t1_valid] = t1_valid_weekly_t1
        else:
            # 回退：直接檢查日頻在 T+1 是否兩檔有價
            for pos_end in wk_pos_valid:
                pos_t1 = pos_end + 1
                if pos_t1 >= len(daily_index):
                    break
                has_x = pd.notna(px_daily.iloc[pos_t1][s1])
                has_y = pd.notna(px_daily.iloc[pos_t1][s2])
                t1_tradeable_flag.iloc[pos_t1] = bool(has_x and has_y)

        # z_t1：將週末 z 移到 T+1（僅在可交易時保留）
        z_t1_daily = pd.Series(np.nan, index=daily_index)
        if include_z_t1:
            z_vals_aligned = z_vals[valid_wk_mask]
            z_vals_t1 = z_vals_aligned[t1_in_range_mask]
            z_t1_daily.iloc[pos_t1_valid] = z_vals_t1
            z_t1_daily = z_t1_daily.where(t1_tradeable_flag, np.nan)

        # 日頻價（raw 與 log）
        px_x_raw = px_daily[s1]
        px_y_raw = px_daily[s2]
        px_x_log = px_daily_log[s1]
        px_y_log = px_daily_log[s2]

        # 組裝輸出（為相容現有 Loader：預設 px_x/px_y 為對數價）
        data_dict = {
            "z": z_daily.values,
            "beta": beta_daily.values,
            "px_x": px_x_log.values,
            "px_y": px_y_log.values,
            "px_x_raw": px_x_raw.values,
            "px_y_raw": px_y_raw.values,
            "week_end_flag": week_end_flag.values,
            "t1_tradeable_flag": t1_tradeable_flag.values,
        }
        if include_alpha_daily:
            data_dict["alpha_daily"] = alpha_daily.values
        if include_z_t1:
            data_dict["z_t1"] = z_t1_daily.values

        df_pair = pd.DataFrame(data_dict, index=daily_index)

        # 輸出檔案（overwrite/覆蓋模式直接重寫）
        df_pair.to_pickle(out_path)
        built += 1
        if built % 100 == 0:
            print(f"[INFO] Built pairs: {built}")

    # 子目錄 meta.json（記錄該組合）
    meta = {
        "version": "v1",
        "L_tag": out_dir.parent.name,         # 例如 L200
        "Z_tag": out_dir.name,                # 例如 Z004
        "formation_length_years": float(formation_length),
        "formation_window_weeks": int(math.ceil(float(formation_length) * 52.0)),
        "z_window_weeks": int(z_window_weeks),
        "ddof": int(ddof),
        "alpha_column": "alpha_daily" if include_alpha_daily else None,
        "include_alpha_daily": bool(include_alpha_daily),
        "include_z_t1": bool(include_z_t1),
        "use_weekly_t1_for_tradeable": bool(use_weekly_t1_for_tradeable),
        "pairs_built": int(built),
        "pairs_skipped": int(skipped),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return built, skipped


def update_manifest(root: Path,
                    prices_path: str,
                    weekly_path: str,
                    entry: dict):
    """更新根目錄的 manifest.json（累積所有已建組合的紀錄）。"""
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            try:
                manifest = json.load(f)
            except Exception:
                manifest = {}
    else:
        manifest = {}

    manifest.setdefault("version", "v1")
    manifest["built_from_prices"] = {"path": prices_path}
    manifest["built_from_weekly"] = {"path": weekly_path}
    manifest.setdefault("builds", [])

    # 若同一個 L_tag/Z_tag 已存在，先移除舊的再新增
    builds = [b for b in manifest["builds"]
              if not (b.get("L_tag") == entry.get("L_tag") and b.get("Z_tag") == entry.get("Z_tag"))]
    builds.append(entry)
    manifest["builds"] = builds

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Build weekly pair cache for multiple L × Z combos.")
    ap.add_argument("--prices", type=str, default="data/prices.pkl", help="Path to daily adjusted close (wide DataFrame).")
    ap.add_argument("--weekly", type=str, default="data/weekly_prices.pkl", help="Path to weekly file with W_Close and T1_Close.")
    ap.add_argument("--pairs", type=str, default="cache/top_pairs_annual.csv", help="CSV with stock1,stock2 or pair.")
    ap.add_argument("--root", type=str, default="cache/rolling_cache_weekly_v1", help="Output cache root.")
    ap.add_argument("--formation-lengths", type=str, required=True, help='Comma floats in years, e.g., "2,2.5,3,3.5,4"')
    ap.add_argument("--z-windows-weeks", type=str, required=True, help='Comma ints in weeks, e.g., "4,8,13,26,52"')
    ap.add_argument("--overwrite-mode", type=str, default="overwrite", choices=["overwrite", "skip", "clean"],
                    help="Existing files policy: overwrite (default), skip existing, or clean the combo folder before build.")
    ap.add_argument("--max-pairs", type=int, default=None, help="Limit number of pairs for a quick build.")
    # 新增參數
    ap.add_argument("--ddof", type=int, default=1, choices=[0, 1], help="ddof for cov/var/std; use the same value for consistency.")
    ap.add_argument("--no-alpha", action="store_true", help="Do not output alpha_daily.")
    ap.add_argument("--no-z-t1", action="store_true", help="Do not output z_t1.")
    ap.add_argument("--t1-flag-from-weekly", action="store_true", default=True,
                    help="Use weekly T1_Close to build t1_tradeable_flag (default True).")
    ap.add_argument("--t1-flag-from-daily", action="store_true",
                    help="Use daily matrix to build t1_tradeable_flag instead of weekly T1_Close.")
    args = ap.parse_args()

    # 解析旗標優先順序
    use_weekly_t1_for_tradeable = True
    if args.t1_flag_from_daily:
        use_weekly_t1_for_tradeable = False
    elif args.t1_flag_from_weekly:
        use_weekly_t1_for_tradeable = True

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading daily prices: {args.prices}")
    px_daily: pd.DataFrame = pd.read_pickle(args.prices)
    if not isinstance(px_daily.index, pd.DatetimeIndex):
        raise TypeError("Daily prices index must be a DatetimeIndex.")
    px_daily = px_daily.sort_index()
    if px_daily.index.tz is not None:
        # 將時區去除，以避免 weekly 對齊時出現 tz mismatch
        px_daily.index = px_daily.index.tz_localize(None)
    px_daily_log = np.log(px_daily.replace(0.0, np.nan))
    print(f"[INFO] Daily matrix: {px_daily.shape}, {px_daily.index.min().date()} -> {px_daily.index.max().date()}")

    print(f"[INFO] Loading weekly prices: {args.weekly}")
    wk: pd.DataFrame = pd.read_pickle(args.weekly)
    if not isinstance(wk.index, pd.DatetimeIndex):
        raise TypeError("Weekly prices index must be a DatetimeIndex.")
    wk = wk.sort_index()
    if wk.index.tz is not None:
        wk.index = wk.index.tz_localize(None)
    print(f"[INFO] Weekly matrix: {wk.shape}, weeks: {wk.index.min().date()} -> {wk.index.max().date()}")

    print(f"[INFO] Loading pairs: {args.pairs}")
    pairs_df = pd.read_csv(args.pairs, encoding="utf-8-sig", low_memory=False)
    pairs_df = to_pair_id_df(pairs_df)
    pairs_df = pairs_df.drop_duplicates(subset=["pair_id"])
    if args.max_pairs:
        pairs_df = pairs_df.head(int(args.max_pairs))
    print(f"[INFO] Unique pairs: {len(pairs_df)}")

    L_list = parse_list_floats(args.formation_lengths)
    Z_list = parse_list_ints(args.z_windows_weeks)

    # 合理性提示：若某些 Z 大於形成窗，提醒但不禁止
    for z in Z_list:
        if z > math.ceil(max(L_list) * 52.0):
            print(f"[WARN] Some z_window_weeks ({z}) > formation_window_weeks; ensure this is intended.")

    print(f"[INFO] Target combos: L={L_list} (years), Z={Z_list} (weeks) -> total {len(L_list)*len(Z_list)} combos")

    total_built = 0
    total_skipped = 0

    for L in L_list:
        L_tag = f"L{int(round(L * 100)):03d}"
        for Z in Z_list:
            Z_tag = f"Z{int(Z):03d}"
            out_dir = root / L_tag / Z_tag

            built, skipped = build_one_combo(
                px_daily=px_daily,
                px_daily_log=px_daily_log,
                wk=wk,
                pairs_df=pairs_df,
                formation_length=L,
                z_window_weeks=Z,
                out_dir=out_dir,
                overwrite_mode=args.overwrite_mode,
                ddof=args.ddof,
                use_weekly_t1_for_tradeable=use_weekly_t1_for_tradeable,
                include_alpha_daily=(not args.no_alpha),
                include_z_t1=(not args.no_z_t1)
            )

            total_built += built
            total_skipped += skipped

            # 更新 manifest
            update_manifest(
                root=root,
                prices_path=args.prices,
                weekly_path=args.weekly,
                entry={
                    "L_tag": L_tag,
                    "Z_tag": Z_tag,
                    "formation_length_years": float(L),
                    "z_window_weeks": int(Z),
                    "ddof": int(args.ddof),
                    "alpha_column": "alpha_daily" if (not args.no_alpha) else None,
                    "include_alpha_daily": bool(not args.no_alpha),
                    "include_z_t1": bool(not args.no_z_t1),
                    "use_weekly_t1_for_tradeable": bool(use_weekly_t1_for_tradeable),
                    "pairs_built": int(built),
                    "pairs_skipped": int(skipped),
                    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
                }
            )

    print(f"[INFO] Done. Built={total_built}, Skipped={total_skipped}, Root={root}")


if __name__ == "__main__":
    main()