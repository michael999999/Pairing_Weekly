#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
將半年度 (YYYYH1/ YYYYH2) 的配對檔轉換為年度：
- 將 YYYYH1 轉為 YYYY（trading_period）
- 移除 YYYYH2
- 將 trading_start / trading_end 固定為該年 01-01 / 12-31（全年）
- 其他欄位維持不變；可對 (trading_period, stock1, stock2) 去重
- 輸出前依 formation_length、trading_period 進行升冪排序
"""

import argparse
from pathlib import Path
import pandas as pd
import re


def convert_pairs_to_annual(
    input_path: str = "cache/top_pairs.csv",
    output_path: str = "cache/top_pairs_annual.csv",
    dedupe: bool = True,
) -> None:
    # 讀取 CSV
    print(f"[INFO] Loading pairs from: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)

    if "trading_period" not in df.columns:
        raise KeyError("Column 'trading_period' is missing in the input CSV.")

    orig_cols = df.columns.tolist()
    n_orig = len(df)
    print(f"[INFO] Rows loaded: {n_orig}")

    # 保留原始 trading_period 以供參考
    df["_source_trading_period"] = df["trading_period"].astype(str)

    # 建立 H1/H2 遮罩
    tp = df["_source_trading_period"]
    h1_mask = tp.str.match(r"^\d{4}H1$", na=False)
    h2_mask = tp.str.match(r"^\d{4}H2$", na=False)

    n_h1 = int(h1_mask.sum())
    n_h2 = int(h2_mask.sum())
    print(f"[INFO] H1 rows: {n_h1}, H2 rows: {n_h2}")

    # 移除 H2
    df_out = df[~h2_mask].copy()

    # H1 → YYYY
    df_out.loc[h1_mask, "trading_period"] = df_out.loc[h1_mask, "_source_trading_period"].str.slice(0, 4)

    # 將所有「年度型」的列（YYYY）之 trading_start / trading_end 設為全年
    # 包含原本就是 YYYY 的、以及剛由 H1 轉換成 YYYY 的
    year_mask = df_out["trading_period"].astype(str).str.match(r"^\d{4}$", na=False)
    if "trading_start" not in df_out.columns:
        df_out["trading_start"] = ""
    if "trading_end" not in df_out.columns:
        df_out["trading_end"] = ""

    years = df_out.loc[year_mask, "trading_period"].astype(str)
    df_out.loc[year_mask, "trading_start"] = years + "-01-01"
    df_out.loc[year_mask, "trading_end"] = years + "-12-31"

    # 去除暫存欄位
    df_out.drop(columns=["_source_trading_period"], inplace=True)

    # 去重：避免同一年同一對重複
    if dedupe and all(c in df_out.columns for c in ["trading_period", "stock1", "stock2"]):
        before = len(df_out)
        df_out = df_out.drop_duplicates(subset=["trading_period", "stock1", "stock2"], keep="first")
        print(f"[INFO] Dedupe (trading_period, stock1, stock2): {before} -> {len(df_out)}")

    # 盡量維持原欄位順序
    out_cols = [c for c in orig_cols if c in df_out.columns] + [c for c in df_out.columns if c not in orig_cols]
    df_out = df_out[out_cols]

    # 排序：依 formation_length、trading_period 升冪
    # - 將 formation_length 嘗試轉成數值（若不存在則警示並忽略）
    # - 將 trading_period 轉為數字年份（YYYY）
    print("[INFO] Sorting by formation_length (asc), then trading_period (asc).")
    if "formation_length" in df_out.columns:
        df_out["_formation_length_sort"] = pd.to_numeric(df_out["formation_length"], errors="coerce")
    else:
        print("[WARN] Column 'formation_length' not found; will sort by trading_period only.")
        df_out["_formation_length_sort"] = 0  # 使排序不因缺列而失敗

    df_out["_year_sort"] = pd.to_numeric(df_out["trading_period"], errors="coerce")

    # 使用穩定排序（mergesort），方便保留相同鍵的原始相對順序
    df_out = (
        df_out.sort_values(by=["_formation_length_sort", "_year_sort"], kind="mergesort")
              .drop(columns=["_formation_length_sort", "_year_sort"])
              .reset_index(drop=True)
    )

    # 輸出
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 報告
    n_out = len(df_out)
    uniq_periods = sorted(df_out["trading_period"].astype(str).unique().tolist()) if "trading_period" in df_out.columns else []
    print(f"[INFO] Saved: {output_path.resolve()}")
    print(f"[INFO] Output rows: {n_out} (dropped H2 rows: {n_h2})")
    print(f"[INFO] Unique trading_periods: {', '.join(uniq_periods[:10])}{' ...' if len(uniq_periods) > 10 else ''}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert half-year trading_period (YYYYH1/YYYYH2) to annual (YYYY), drop H2, set full-year dates, and sort."
    )
    parser.add_argument("--input", type=str, default="cache/top_pairs.csv", help="Path to input CSV")
    parser.add_argument("--output", type=str, default="cache/top_pairs_annual.csv", help="Path to output CSV")
    parser.add_argument("--no-dedupe", action="store_true", help="Do not drop duplicates on (trading_period, stock1, stock2)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_pairs_to_annual(
        input_path=args.input,
        output_path=args.output,
        dedupe=not args.no_dedupe,
    )