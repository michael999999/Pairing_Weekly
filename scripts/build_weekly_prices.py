#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
將日頻 adjusted close 轉為週頻資料，產出單一檔 weekly_prices.pkl，
欄位為 MultiIndex: [Symbol, {W_Close, T1_Close}]

規則：
- 週界定：W-FRI（以週五為結束；若週五休市則該週最後一個交易日）
- W_Close：每檔在該週內「最後一個有價日」的收盤（不跨週 forward fill）
- T1_Close：以該「最後一個有價日」後的「下一個有價日」收盤（每檔獨立計算）
- 若整週無價，W_Close 與 T1_Close 皆為 NaN
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_weekly_prices(
    input_path: str = "data/prices.pkl",
    output_path: str = "data/weekly_prices.pkl",
    week_freq: str = "W-FRI",
) -> None:
    # 讀取資料
    print(f"[INFO] Loading daily prices from: {input_path}")
    prices: pd.DataFrame = pd.read_pickle(input_path)

    # 基本檢查與整理
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame index must be a DatetimeIndex.")

    prices = prices.sort_index()
    # 嘗試將所有欄位轉為數值（非數值轉 NaN）
    prices = prices.apply(pd.to_numeric, errors="coerce")
    symbols = prices.columns
    n_syms = len(symbols)
    n_days = len(prices.index)

    print(f"[INFO] Daily matrix shape: {prices.shape} (days x symbols)")
    print(f"[INFO] Date range: {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"[INFO] Number of symbols: {n_syms}")

    # 預先計算「下一個有價日」的收盤矩陣（每檔）
    # 作法：先 shift(-1) 取「隔日」價，再用 bfill() 沿時間向後填補，得到「下一個非 NaN 的未來價」
    # 若已是樣本最後一天，或之後皆無價，則維持 NaN
    next_after = prices.shift(-1).bfill()

    # 為了把週內「最後一個有價日」轉成全域（日頻）的位置索引，先建立日期->位置的對應
    daily_index = prices.index
    # 建立加速查找：把週內相對位置轉為全域位置
    # 直接透過 get_indexer(週內日期序列) 取得全域位置陣列，再以該陣列做索引
    next_after_values = next_after.values  # 形狀：(n_days, n_syms)

    # 逐週處理：取每週各檔「最後一個有價日」的收盤（W_Close）
    # 並據此映射出 T1_Close（下一個有價日的收盤）
    weekly_labels = []
    wclose_rows = []
    t1_rows = []

    # 使用 Grouper 以 W-FRI 切週
    grouper = pd.Grouper(freq=week_freq)

    print(f"[INFO] Building weekly bars with freq={week_freq} ...")
    for week_end, g in prices.groupby(grouper, sort=True):
        # g: 該週的日頻切片（可能 0~5 個交易日，跨市可能更多）
        if len(g) == 0:
            continue  # 空週（理論上不會出現）

        # 將該週的日頻資料轉成 numpy 陣列，便於向量化運算
        arr = g.values  # 形狀：(m_days_in_week, n_syms)
        m = arr.shape[0]
        mask = ~np.isnan(arr)  # True 表示該日該檔有價
        has_valid = mask.any(axis=0)  # 各檔該週是否有任一有價日（長度 n_syms）

        # 找出每檔「最後一個有價日」在該週內的相對位置
        # 技巧：反轉後取 argmax，可得到最後一個 True 的位置
        pos_rev = np.argmax(mask[::-1, :], axis=0)  # 若全 False，值為 0（需用 has_valid 篩掉）
        last_pos = (m - 1) - pos_rev
        last_pos[~has_valid] = -1  # 無有效價的檔，以 -1 標記（避免誤用）

        # 構造 W_Close（每檔週末價）：取最後一個有價日的收盤
        wclose = np.full(n_syms, np.nan, dtype=np.float64)
        valid_idx = np.where(has_valid)[0]
        if valid_idx.size > 0:
            wclose[valid_idx] = arr[last_pos[valid_idx], valid_idx]

        # 構造 T1_Close：以「最後一個有價日」之後的「下一個有價日」收盤
        # 先將該週的日頻索引轉為全域（日頻）位置索引
        week_global_pos = daily_index.get_indexer(g.index)  # 形狀：(m,)
        # 取出每檔對應的全域位置
        t1 = np.full(n_syms, np.nan, dtype=np.float64)
        if valid_idx.size > 0:
            last_global_pos = week_global_pos[last_pos[valid_idx]]
            t1[valid_idx] = next_after_values[last_global_pos, valid_idx]

        weekly_labels.append(week_end)
        wclose_rows.append(wclose)
        t1_rows.append(t1)

    # 組裝成 DataFrame（週頻）
    weekly_index = pd.DatetimeIndex(weekly_labels, name="WeekEnd")
    w_close_df = pd.DataFrame(np.vstack(wclose_rows), index=weekly_index, columns=symbols)
    t1_close_df = pd.DataFrame(np.vstack(t1_rows), index=weekly_index, columns=symbols)

    # 合併為 MultiIndex 欄位：[Symbol, {W_Close, T1_Close}]
    combined = pd.concat([w_close_df, t1_close_df], axis=1, keys=["W_Close", "T1_Close"])
    combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
    combined.columns.set_names(["Symbol", "Field"], inplace=True)

    # 儲存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(combined, output_path)

    # 基本統計與提示
    n_weeks = combined.shape[0]
    print(f"[INFO] Weekly matrix shape: {combined.shape} (weeks x 2*symbols)")
    print(f"[INFO] Weeks range: {combined.index.min().date()} -> {combined.index.max().date()}")
    print(f"[INFO] Saved weekly prices with W_Close and T1_Close to: {output_path.resolve()}")
    # 額外提醒：最後一週通常 T1_Close 為 NaN（因資料結尾無下一交易日）
    last_week = combined.index.max()
    n_t1_nan_last = combined.xs("T1_Close", axis=1, level="Field").loc[last_week].isna().sum()
    print(f"[INFO] Sanity: On the last week ({last_week.date()}), T1_Close NaNs across symbols: {n_t1_nan_last}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build weekly close (W_Close) and next trading day close (T1_Close) from daily adjusted close."
    )
    parser.add_argument("--input", type=str, default="data/prices.pkl", help="Path to daily prices .pkl")
    parser.add_argument("--output", type=str, default="data/weekly_prices.pkl", help="Path to output weekly .pkl")
    parser.add_argument("--freq", type=str, default="W-FRI", help="Weekly frequency label (default: W-FRI)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_weekly_prices(input_path=args.input, output_path=args.output, week_freq=args.freq)