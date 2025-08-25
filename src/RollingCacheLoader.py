# -*- coding: utf-8 -*-
"""
RollingCacheLoader
- 讀取週頻 pair-cache（多組 formation_length × z_window_weeks）
- 與 backtest_full.py 相容：load_panel(fields=("z",| "beta" | "px",), ...) 與 check_missing(pair_ids)
- 欄位回傳格式：MultiIndex columns = [Field, pair_id]
  - "z"    -> panel["z"]           -> DataFrame[dates × pairs]
  - "beta" -> panel["beta"]        -> DataFrame[dates × pairs]
  - "px"   -> panel["px_x"], panel["px_y"]（依 price_type 選 log/raw）
- 註解：繁體中文；log 為英文
"""

import os
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import pandas as pd
import numpy as np


class RollingCacheLoader:
    """
    週頻 pair-cache 載入器（與 backtest_full.py 相容）
    - root: cache 根目錄（例如 cache/rolling_cache_weekly_v1）
    - formation_length: 年（float，例如 2.0 -> L200）
    - z_window: 週數（int，例如 4 -> Z004）
    - price_type: "log" 或 "raw"
    """

    def __init__(self,
                 root: str,
                 price_type: str = "log",
                 formation_length: float = 0.5,
                 z_window: int = 104,
                 log_level: str = "INFO") -> None:
        self.root = Path(root)
        self.price_type = str(price_type).lower()
        self.formation_length = float(formation_length)
        self.z_window = int(z_window)

        # 目錄標籤：L{years*100:03d}/Z{weeks:03d}
        self.L_tag = f"L{int(round(self.formation_length * 100)):03d}"
        self.Z_tag = f"Z{int(self.z_window):03d}"
        self.base_dir = self.root / self.L_tag / self.Z_tag / "pairs"

        # logger
        self.logger = logging.getLogger(f"RollingCacheLoader[{self.L_tag}/{self.Z_tag}]")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            h.setFormatter(fmt)
            self.logger.addHandler(h)
        self.logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))

        # 快取（避免重複讀檔）：pair_id -> DataFrame
        self._pair_cache: Dict[str, pd.DataFrame] = {}

        # 基本檢查（目錄存在）
        if not self.base_dir.exists():
            self.logger.warning(f"Base dir not found: {self.base_dir}")

        # 欄位映射（px_x/px_y 使用 log 或 raw）
        self._px_fields = ("px_x", "px_y") if self.price_type == "log" else ("px_x_raw", "px_y_raw")

        self.logger.info(f"Init loader: root={self.root} L={self.L_tag} Z={self.Z_tag} price_type={self.price_type}")

    # 內部工具 ---------------------------------------------------------------

    def _pair_path(self, pair_id: str) -> Path:
        """取得 pair 檔案路徑。"""
        return self.base_dir / f"{pair_id}.pkl"

    def _load_pair_df(self, pair_id: str) -> Optional[pd.DataFrame]:
        """讀取單一 pair 的 DataFrame；失敗回傳 None。"""
        if pair_id in self._pair_cache:
            return self._pair_cache[pair_id]

        path = self._pair_path(pair_id)
        if not path.exists():
            return None

        try:
            df = pd.read_pickle(path)
            # 基本清理：確保索引為 DatetimeIndex、排序
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.sort_index()
            self._pair_cache[pair_id] = df
            return df
        except Exception as e:
            self.logger.error(f"Failed to load pair cache: {pair_id} -> {e}")
            return None

    @staticmethod
    def _clip_date_range(idx: pd.DatetimeIndex, date_range: Optional[Tuple[str, str]]) -> slice:
        """由 (start, end) 建立 loc 的切片；若無提供則回傳全範圍。"""
        if not date_range or (not date_range[0] and not date_range[1]):
            return slice(None)
        start = pd.to_datetime(date_range[0]) if date_range[0] else None
        end = pd.to_datetime(date_range[1]) if date_range[1] else None
        return slice(start, end)

    # 公開介面 ---------------------------------------------------------------

    def check_missing(self, pair_ids: Iterable[str]) -> List[str]:
        """回傳缺少快取檔案的 pair_ids。"""
        missing = []
        for pid in pair_ids:
            if not self._pair_path(pid).exists():
                missing.append(pid)
        return missing

    def load_panel(self,
                   pair_ids: Iterable[str],
                   fields: Tuple[str, ...] = ("z",),
                   date_range: Optional[Tuple[str, str]] = None,
                   join: str = "outer",
                   allow_missing: bool = True) -> pd.DataFrame:
        """
        載入多個 pair 的欄位，組合為單一 DataFrame（MultiIndex columns=[Field, pair_id]）。
        - fields: 可含 "z"、"beta"、"px"
          - "px" 代表同時輸出 px_x 與 px_y（依 price_type 選 log/raw 欄位）
        - date_range: (start, end) 皆包含；None 表示全區間
        - join: "outer"（預設）或 "inner"，控制跨 pair 的日期索引對齊方式
        - allow_missing: True 則跳過不存在的 pair；False 則遇缺檔直接 raise
        """
        # 正規化 fields
        req = []
        for f in fields:
            f = str(f).lower()
            if f not in ("z", "beta", "px"):
                raise ValueError(f"Unsupported field: {f}")
            req.append(f)
        fields = tuple(req)

        # 收集每個欄位對應的 pair->Series
        per_field_series: Dict[str, Dict[str, pd.Series]] = {}
        per_field_index: Dict[str, pd.DatetimeIndex] = {}

        # 準備切片
        date_slice = None  # 延後於第一支讀入後決定 loc
        # 逐 pair 讀入
        loaded_count = 0
        for pid in pair_ids:
            df = self._load_pair_df(pid)
            if df is None:
                if allow_missing:
                    self.logger.debug(f"Skip missing pair: {pid}")
                    continue
                raise FileNotFoundError(f"Pair cache not found: {pid}")

            # 決定日期切片（使用第一支的索引建立 loc 範圍）；後續用 .loc 以 Timestamp 切
            if date_slice is None:
                date_slice = self._clip_date_range(df.index, date_range)

            for f in fields:
                if f == "z":
                    ser = df.loc[date_slice, "z"] if "z" in df.columns else pd.Series(np.nan, index=df.index)
                    per_field_series.setdefault("z", {})[pid] = ser
                elif f == "beta":
                    ser = df.loc[date_slice, "beta"] if "beta" in df.columns else pd.Series(np.nan, index=df.index)
                    per_field_series.setdefault("beta", {})[pid] = ser
                elif f == "px":
                    # 價格欄位：依 price_type 選 log 或 raw
                    f_x, f_y = self._px_fields  # ("px_x","px_y") 或 ("px_x_raw","px_y_raw")
                    sx = df.loc[date_slice, f_x] if f_x in df.columns else pd.Series(np.nan, index=df.index)
                    sy = df.loc[date_slice, f_y] if f_y in df.columns else pd.Series(np.nan, index=df.index)
                    per_field_series.setdefault(f_x, {})[pid] = sx
                    per_field_series.setdefault(f_y, {})[pid] = sy

            loaded_count += 1

        if loaded_count == 0:
            # 回傳空 DataFrame（與回測引擎相容）
            return pd.DataFrame()

        # 組裝各欄位的矩陣
        panels: List[pd.DataFrame] = []
        for key, mapping in per_field_series.items():
            if not mapping:
                continue
            # 將所有 pair 的 index 以 join 方式對齊
            if join == "inner":
                # 交集
                idx = None
                for s in mapping.values():
                    idx = s.index if idx is None else idx.intersection(s.index)
            else:
                # 聯集（預設）
                idx = pd.Index([])
                for s in mapping.values():
                    idx = idx.union(s.index)
            idx = idx.sort_values()

            # 以 union/inner 的索引對齊，欄序按輸入 pair_ids 順序
            cols = [pid for pid in pair_ids if pid in mapping]
            data = {pid: mapping[pid].reindex(idx) for pid in cols}
            df_field = pd.DataFrame(data, index=idx)

            panels.append(pd.concat({key: df_field}, axis=1))

        if not panels:
            return pd.DataFrame()

        panel = pd.concat(panels, axis=1)

        # 設定 MultiIndex 欄位名稱（利於 panel_z["z"]等操作）
        try:
            panel.columns.set_names(["Field", "pair_id"], inplace=True)
        except Exception:
            pass

        return panel