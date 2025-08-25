#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
以週收盤對數價進行滾動 OLS 與 z-score 計算，並將 β 對齊至日頻（T+1 起 forward-fill），
可一次建立多個 formation_length × z_window_weeks 的週頻 pair-cache。

輸入：
- data/prices.pkl（日頻 adjusted close，寬表）
- data/weekly_prices.pkl（週末 W_Close 與下一交易日 T1_Close；MultiIndex columns: [Symbol, {W_Close, T1_Close}]）
- cache/top_pairs_annual.csv（stock1, stock2 或 pair 欄位）

輸出（按組合分層）：
- cache/rolling_cache_weekly_v1/L{L_tag}/Z{zwin}/pairs/{pair_id}.pkl（DataFrame，index=日頻）
- cache/rolling_cache_weekly_v1/L{L_tag}/Z{zwin}/meta.json（該組合的建置資訊）
- cache/rolling_cache_weekly_v1/manifest.json（累積記錄所有已建組合）

參數：
- --formation-lengths：以年為單位，逗號分隔（例：2,2.5,3,3.5,4）
- --z-windows-weeks：以週為單位，逗號分隔（例：4,8,13,26,52）
- --overwrite-mode：overwrite（預設）、skip（遇既有檔則跳過）、clean（先清除該 L/Z 目錄再重建）
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import shutil
from datetime import datetime


def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def to_pair_id_df(df: pd.DataFrame) -> pd.DataFrame:
    """保證 pair_id 欄位存在；若只有 pair 字串，解析為 stock1/stock2。"""
    if "stock1" in df.columns and "stock2" in df.columns:
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
        df["stock1"] = tmp.apply(lambda t: t[0])
        df["stock2"] = tmp.apply(lambda t: t[1])
        df["pair_id"] = df["stock1"].astype(str) + "__" + df["stock2"].astype(str)
        return df
    raise ValueError("Input CSV must contain 'stock1' and 'stock2' or 'pair' column.")


def rolling_ols_y_on_x(y: pd.Series, x: pd.Series, win: int) -> Tuple[pd.Series, pd.Series]:
    """用週頻對數價執行滾動 OLS（y=alpha+beta*x），回傳 alpha, beta（只在滿足窗口時有值）。"""
    y = y.copy()
    x = x.reindex_like(y)
    mean_x = x.rolling(win, min_periods=win).mean()
    mean_y = y.rolling(win, min_periods=win).mean()
    cov_xy = x.rolling(win, min_periods=win).cov(y)
    var_x = x.rolling(win, min_periods=win).var(ddof=0)
    beta = cov_xy / var_x.replace(0.0, np.nan)
    alpha = mean_y - beta * mean_x
    return alpha, beta

def rolling_zscore(s: pd.Series, win: int) -> pd.Series:
    """以窗口 win（週數）計算滾動 z-score。"""
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


def build_one_combo(px_daily: pd.DataFrame,
                    px_daily_log: pd.DataFrame,
                    wk: pd.DataFrame,
                    pairs_df: pd.DataFrame,
                    formation_length: float,
                    z_window_weeks: int,
                    out_dir: Path,
                    overwrite_mode: str = "overwrite") -> Tuple[int, int]:
    """建立單一 L × Z 組合的所有 pair 快取，回傳 (built, skipped)。"""
    daily_index = px_daily.index

    # 週末索引與 W_Close 存取輔助
    if not isinstance(wk.columns, pd.MultiIndex) or "W_Close" not in wk.columns.get_level_values(1):
        raise ValueError("weekly file must be a MultiIndex columns with Field including 'W_Close'.")
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
    print(f"[INFO] Building combo L={formation_length}y (form={win_weeks_form}w), Z={win_weeks_z}w -> {out_dir}")

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

        # 價格可用性
        if s1 not in px_daily.columns or s2 not in px_daily.columns:
            print(f"[WARN] Skip {pid}: symbol missing in daily prices.")
            skipped += 1
            continue
        try:
            x_w = wk[(s1, "W_Close")].copy()
            y_w = wk[(s2, "W_Close")].copy()
        except KeyError:
            print(f"[WARN] Skip {pid}: W_Close not found for one of symbols in weekly file.")
            skipped += 1
            continue

        # 以週頻對數價做 OLS 與殘差 z（窗口：週）
        x_w_log = np.log(x_w.replace(0.0, np.nan))
        y_w_log = np.log(y_w.replace(0.0, np.nan))
        alpha_w, beta_w = rolling_ols_y_on_x(y_w_log, x_w_log, win_weeks_form)
        spread_w = y_w_log - (alpha_w + beta_w * x_w_log)
        z_w = rolling_zscore(spread_w, win_weeks_z)

        # 對齊到日頻
        week_end_flag = pd.Series(False, index=daily_index)
        week_end_flag.iloc[wk_pos_valid] = True

        # z：僅在週末日填值（注意使用 values 對齊 valid mask，避免索引對不上的錯誤）
        z_daily = pd.Series(np.nan, index=daily_index)
        z_vals = z_w.values  # 與 week_end_index 對齊
        z_daily.iloc[wk_pos_valid] = z_vals[valid_wk_mask]

        # β（日頻有效）：每個週末估計，從 T+1 起 forward-fill 到下一次週末前
        beta_daily = pd.Series(np.nan, index=daily_index)
        beta_vals = beta_w.values  # 與 week_end_index 對齊
        for i in range(len(wk_pos_valid)):
            pos_end = wk_pos_valid[i]
            pos_t1 = pos_end + 1
            if pos_t1 >= len(daily_index):
                break  # 沒有下一個交易日
            # 下一個週末位置
            next_pos_end = wk_pos_valid[i + 1] if (i + 1) < len(wk_pos_valid) else len(daily_index)
            # 該區間用同一個 beta（可能為 NaN，表示窗口不足）
            beta_val = beta_vals[valid_wk_mask][i]
            beta_daily.iloc[pos_t1:next_pos_end] = beta_val

        # T+1 可交易旗標：在 T+1 當日兩檔皆有收盤價
        t1_tradeable_flag = pd.Series(False, index=daily_index)
        for pos_end in wk_pos_valid:
            pos_t1 = pos_end + 1
            if pos_t1 >= len(daily_index):
                break
            has_x = pd.notna(px_daily.iloc[pos_t1][s1])
            has_y = pd.notna(px_daily.iloc[pos_t1][s2])
            t1_tradeable_flag.iloc[pos_t1] = bool(has_x and has_y)

        # 日頻價（raw 與 log）
        px_x_raw = px_daily[s1]
        px_y_raw = px_daily[s2]
        px_x_log = px_daily_log[s1]
        px_y_log = px_daily_log[s2]

        # 組裝輸出（為相容現有 Loader：預設 px_x/px_y 為對數價）
        df_pair = pd.DataFrame({
            "z": z_daily.values,
            "beta": beta_daily.values,
            "px_x": px_x_log.values,
            "px_y": px_y_log.values,
            "px_x_raw": px_x_raw.values,
            "px_y_raw": px_y_raw.values,
            "week_end_flag": week_end_flag.values,
            "t1_tradeable_flag": t1_tradeable_flag.values,
        }, index=daily_index)

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
        "pairs_built": int(built),
        "pairs_skipped": int(skipped),
        "built_at": datetime.utcnow().isoformat() + "Z"
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
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading daily prices: {args.prices}")
    px_daily: pd.DataFrame = pd.read_pickle(args.prices)
    if not isinstance(px_daily.index, pd.DatetimeIndex):
        raise TypeError("Daily prices index must be a DatetimeIndex.")
    px_daily = px_daily.sort_index()
    px_daily_log = np.log(px_daily.replace(0.0, np.nan))
    print(f"[INFO] Daily matrix: {px_daily.shape}, {px_daily.index.min().date()} -> {px_daily.index.max().date()}")

    print(f"[INFO] Loading weekly prices: {args.weekly}")
    wk: pd.DataFrame = pd.read_pickle(args.weekly)
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
                overwrite_mode=args.overwrite_mode
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
                    "pairs_built": int(built),
                    "pairs_skipped": int(skipped),
                    "built_at": datetime.utcnow().isoformat() + "Z"
                }
            )

    print(f"[INFO] Done. Built={total_built}, Skipped={total_skipped}, Root={root}")


if __name__ == "__main__":
    main()