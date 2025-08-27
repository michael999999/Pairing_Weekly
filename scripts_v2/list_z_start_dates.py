#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
從指定 root（預設：cache/rolling_cache_weekly_v2）遞迴掃描所有 L*/Z*/pairs，
列出每個 pair 的 z-score 可用起始日期，並依起始日期排序輸出。
- 預設以週末欄位 z 作為起始（第一個非 NaN 的日期）
- 可用 --column z_t1 以 T+1 的 z-score 起始日期排序
- 可使用 --l-filter / --z-filter 篩選特定 L_tag / Z_tag（支援萬用字元）
- 可選擇輸出成 CSV 檔

程式輸出（print/log）：英文
程式碼註解：繁體中文
"""

import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import fnmatch

import pandas as pd


# ====== 掃描與檔案偵測 ======

def find_pairs_dirs(root: Path, l_filter: str = "*", z_filter: str = "*") -> List[Tuple[str, str, Path]]:
    """從 root 下尋找所有 L*/Z*/pairs 目錄，回傳清單 (L_tag, Z_tag, pairs_dir)。"""
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    results: List[Tuple[str, str, Path]] = []
    for L_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("L")]):
        if not fnmatch.fnmatch(L_dir.name, l_filter):
            continue
        for Z_dir in sorted([p for p in L_dir.iterdir() if p.is_dir() and p.name.startswith("Z")]):
            if not fnmatch.fnmatch(Z_dir.name, z_filter):
                continue
            pairs_dir = Z_dir / "pairs"
            if pairs_dir.is_dir():
                # 確認底下有 pkl 才收
                if any(p.suffix == ".pkl" for p in pairs_dir.iterdir()):
                    results.append((L_dir.name, Z_dir.name, pairs_dir))
    return results


# ====== 工具函式 ======

def first_valid_timestamp(df: pd.DataFrame, col: str) -> Optional[pd.Timestamp]:
    """回傳指定欄位第一個非 NaN 的索引（Timestamp）；若不存在則回傳 None。"""
    if col not in df.columns:
        return None
    idx = df[col].first_valid_index()
    if idx is None:
        return None
    if not isinstance(idx, pd.Timestamp):
        try:
            idx = pd.to_datetime(idx)
        except Exception:
            return None
    return idx


def format_ts(ts: Optional[pd.Timestamp]) -> str:
    """將 Timestamp 格式化為 YYYY-MM-DD；None 則回傳空字串。"""
    if ts is None:
        return ""
    try:
        return ts.date().isoformat()
    except Exception:
        return str(ts)


def collect_start_dates_for_pairs_dir(pairs_dir: Path, chosen_col: str,
                                      L_tag: str, Z_tag: str) -> List[Dict[str, Any]]:
    """掃描單一 pairs 目錄下所有 pkl，蒐集各 pair 的起始日期資訊。"""
    results: List[Dict[str, Any]] = []
    pkl_files = sorted([p for p in pairs_dir.glob("*.pkl")])

    print(f"[INFO] Scanning pairs dir: {pairs_dir} (files={len(pkl_files)})")

    for pkl_path in pkl_files:
        pid = pkl_path.stem  # 檔名即 pair_id
        try:
            df = pd.read_pickle(pkl_path)
        except Exception as e:
            print(f"[WARN] Failed to read {pkl_path.name}: {e}")
            results.append({
                "L_tag": L_tag,
                "Z_tag": Z_tag,
                "pair_id": pid,
                "first_z_date": None,
                "first_z_t1_date": None,
                "sort_date": None,
                "error": str(e),
            })
            continue

        first_z = first_valid_timestamp(df, "z")
        first_z_t1 = first_valid_timestamp(df, "z_t1")

        sort_date = first_z_t1 if chosen_col == "z_t1" else first_z

        results.append({
            "L_tag": L_tag,
            "Z_tag": Z_tag,
            "pair_id": pid,
            "first_z_date": first_z,
            "first_z_t1_date": first_z_t1,
            "sort_date": sort_date,
            "error": None,
        })

    return results


# ====== 主流程 ======

def main():
    ap = argparse.ArgumentParser(description="List available z-score start dates for all pairs under a cache root and sort by start date.")
    ap.add_argument("--root", type=str, default="cache/rolling_cache_weekly_v2",
                    help="Cache root to scan. Default: cache/rolling_cache_weekly_v2")
    ap.add_argument("--column", type=str, default="z", choices=["z", "z_t1"],
                    help="Which column to use for start date. Default: z")
    ap.add_argument("--l-filter", type=str, default="*",
                    help="Wildcard filter for L_tag folders (e.g., L300, L3*). Default: *")
    ap.add_argument("--z-filter", type=str, default="*",
                    help="Wildcard filter for Z_tag folders (e.g., Z013, Z0*). Default: *")
    ap.add_argument("--output-csv", type=str, default=None, help="Optional path to save results as CSV.")
    ap.add_argument("--top", type=int, default=None, help="Show only top N rows after sorting.")
    ap.add_argument("--desc", action="store_true", help="Sort in descending order.")
    ap.add_argument("--drop-missing", action="store_true", help="Drop pairs with no available start date for the chosen column.")
    args = ap.parse_args()

    root = Path(args.root)
    combos = find_pairs_dirs(root, l_filter=args.l_filter, z_filter=args.z_filter)
    if not combos:
        print(f"[WARN] No L/Z combos found under: {root} with filters L={args.l_filter}, Z={args.z_filter}")
        return

    print(f"[INFO] Root: {root}, combos found: {len(combos)}")

    records: List[Dict[str, Any]] = []
    for L_tag, Z_tag, pairs_dir in combos:
        records.extend(collect_start_dates_for_pairs_dir(pairs_dir, args.column, L_tag, Z_tag))

    # 排序：將 None 排到最後（ascending），或最前（descending）
    def sort_key(rec):
        ts = rec["sort_date"]
        return (ts is None, ts)

    records.sort(key=sort_key, reverse=args.desc)

    # 過濾缺失（可選）
    if args.drop_missing:
        records = [r for r in records if r["sort_date"] is not None]

    # 若設定 top，截取前 N 筆
    if args.top is not None and args.top > 0:
        records = records[:args.top]

    # 輸出（英語）
    print(f"[INFO] Sorted by: {args.column} (descending={bool(args.desc)}), total records: {len(records)}")
    print("L_tag,Z_tag,pair_id,first_z_date,first_z_t1_date,sort_col,sort_date")
    for r in records:
        print(",".join([
            r["L_tag"],
            r["Z_tag"],
            r["pair_id"],
            format_ts(r["first_z_date"]),
            format_ts(r["first_z_t1_date"]),
            args.column,
            format_ts(r["sort_date"]),
        ]))

    # CSV 輸出（如指定）
    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame([
            {
                "L_tag": r["L_tag"],
                "Z_tag": r["Z_tag"],
                "pair_id": r["pair_id"],
                "first_z_date": format_ts(r["first_z_date"]),
                "first_z_t1_date": format_ts(r["first_z_t1_date"]),
                "sort_col": args.column,
                "sort_date": format_ts(r["sort_date"]),
                "error": r.get("error"),
            }
            for r in records
        ])
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved CSV: {out_path}")


if __name__ == "__main__":
    main()