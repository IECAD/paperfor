"""Data provisioning for the UWB benchmark."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..paths import package_path
from .adapters import InputAdapterManager, build_default_manager
from .filters import UWBFilter

DEFAULT_RAW_DIR = package_path("1_data", "raw")
DEFAULT_PROCESSED_DIR = package_path("1_data", "processed")
DEFAULT_ENVIRONMENT_MAP = {
    0: 0,  # Empty lecture room
    1: 1,  # Static obstacles (measured)
    2: 2,  # Dynamic obstacles (measured)
    3: 1,  # Static CAD augmentation -> static
    4: 2,  # Dynamic CAD augmentation -> dynamic
}


@dataclass
class SplitData:
    X: np.ndarray
    y_pos: np.ndarray
    y_class: np.ndarray
    meta: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class DataBundle:
    train: SplitData
    val: Optional[SplitData]
    test: SplitData
    scaler: StandardScaler
    label_encoder: LabelEncoder
    metadata: Dict[str, object]
    adapters: InputAdapterManager
    feature_store: Dict[str, Dict[str, np.ndarray]]


class DataProvider:
    """Construct train/val/test splits with consistent preprocessing."""

    def __init__(
        self,
        raw_dir: Path | str = DEFAULT_RAW_DIR,
        processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
        cache_name: str = "dataset_cache",
        filter_type: str = "combo",
        anchors: Tuple[int, ...] = (1, 2, 3, 4),
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        environment_map: Optional[Dict[int, int]] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.cache_name = cache_name
        self.filter_type = filter_type
        self.anchors = anchors
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.environment_map = (environment_map or DEFAULT_ENVIRONMENT_MAP).copy()

        self.cache_path = self.processed_dir / f"{cache_name}.npz"
        self.scaler_path = self.processed_dir / f"{cache_name}_scaler.pkl"
        self.encoder_path = self.processed_dir / f"{cache_name}_encoder.pkl"
        self.meta_path = self.processed_dir / f"{cache_name}_meta.json"

        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def prepare_once(self, force: bool = False) -> DataBundle:
        if not force and self.cache_path.exists() and self.scaler_path.exists() and self.encoder_path.exists():
            return self._load_cache()

        raw_data = self._load_raw()
        if raw_data.empty:
            raise FileNotFoundError(
                f"No raw Excel data found in {self.raw_dir}. Place source files before running."
            )

        train_raw, val_raw, test_raw = self._safe_point_wise_split(raw_data)
        train_df = self._apply_filter(train_raw)
        val_df = self._apply_filter(val_raw) if not val_raw.empty else val_raw
        test_df = self._apply_filter(test_raw)

        scaler = StandardScaler()
        encoder = LabelEncoder()

        train_split = self._build_split(train_df, scaler=scaler, encoder=encoder, fit_scaler=True, fit_encoder=True)
        val_split = self._build_split(val_df, scaler=scaler, encoder=encoder) if not val_df.empty else None
        test_split = self._build_split(test_df, scaler=scaler, encoder=encoder)

        metadata = {
            "train_size": len(train_split.X),
            "val_size": len(val_split.X) if val_split else 0,
            "test_size": len(test_split.X),
            "input_dim": train_split.X.shape[1],
            "anchors": list(self.anchors),
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "filter_type": self.filter_type,
        }

        adapters, feature_store = self._build_feature_variants(train_split, val_split, test_split)
        self._store_cache(train_split, val_split, test_split, scaler, encoder, metadata)
        return DataBundle(train_split, val_split, test_split, scaler, encoder, metadata, adapters, feature_store)

    def _load_raw(self) -> pd.DataFrame:
        if not self.raw_dir.exists():
            self.raw_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        for path in sorted(self.raw_dir.glob("*.xlsx")):
            df = pd.read_excel(path)
            df["source_file"] = path.name
            frames.append(df)
        # Compatibility with legacy layout
        legacy_dir = Path("uwb_data")
        if not frames and legacy_dir.exists():
            for path in sorted(legacy_dir.glob("*.xlsx")):
                df = pd.read_excel(path)
                df["source_file"] = path.name
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        data = pd.concat(frames, ignore_index=True)
        if {"x", "y"}.issubset(data.columns) and "x_m" not in data.columns:
            data["x_m"] = data["x"].astype(float) * 0.96
            data["y_m"] = data["y"].astype(float) * 1.02
        if "nlos_type" not in data.columns:
            data["nlos_type"] = 0
        data = self._remap_environment_labels(data)
        return data

    def _remap_environment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw environment labels to the consolidated (0,1,2) set."""
        if "nlos_type" not in df.columns or not self.environment_map:
            return df

        def _map(value):
            try:
                return self.environment_map.get(int(value), int(value))
            except (TypeError, ValueError):
                return value

        df["nlos_type"] = df["nlos_type"].apply(_map).astype(np.int32)
        return df

    def _apply_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if self.filter_type == "combo":
            return UWBFilter.combo_filter(df)
        if self.filter_type == "snr_weighted":
            return UWBFilter.snr_weighted_filter(df)
        return df

    def _safe_point_wise_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_parts, val_parts, test_parts = [], [], []
        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        sort_candidates = [c for c in ["timestamp", "group", "sequence"] if c in data.columns]
        for env in sorted(data["nlos_type"].unique()):
            env_df = data[data["nlos_type"] == env]
            for point_id in env_df["point_id"].unique():
                point_df = env_df[env_df["point_id"] == point_id]
                if sort_candidates:
                    point_df = point_df.sort_values(sort_candidates)
                else:
                    point_df = point_df.sort_index()

                n_total = len(point_df)
                if n_total == 0:
                    continue
                train_end = max(1, int(n_total * self.train_ratio))
                val_end = max(train_end + 1, int(n_total * (self.train_ratio + self.val_ratio)))
                val_end = min(val_end, n_total - 1) if n_total > 2 else n_total

                train_parts.append(point_df.iloc[:train_end])
                if val_end > train_end:
                    val_parts.append(point_df.iloc[train_end:val_end])
                test_slice = point_df.iloc[val_end:] if val_end < n_total else point_df.iloc[-1:]
                test_parts.append(test_slice)

        def _merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return _merge(train_parts), _merge(val_parts), _merge(test_parts)

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        expected = {
            "DIST": 0.0,
            "RSSI": 0.0,
            "SEQ": 0.0,
            "LOSS": 0.0,
            "CIR": 0.0,
            "FP2": 0.0,
            "MaxNoise": 0.0,
            "SNR": 0.0,
        }
        augmented = df.copy()
        for col, default in expected.items():
            if col not in augmented.columns:
                augmented[col] = default
        return augmented

    def _build_split(
        self,
        df: pd.DataFrame,
        *,
        scaler: StandardScaler,
        encoder: LabelEncoder,
        fit_scaler: bool = False,
        fit_encoder: bool = False,
    ) -> SplitData:
        if df.empty:
            empty_meta = {
                "anchor_id": np.zeros((0,), dtype=np.int32),
                "group_index": np.zeros((0,), dtype=np.int32),
            }
            return SplitData(
                np.zeros((0, 20), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                empty_meta,
            )

        augmented = self._ensure_columns(df)
        feature_matrix, positions, classes, sample_meta = self._create_anchor_windows(augmented)
        feature_matrix = self._fill_nan(feature_matrix)

        if fit_scaler:
            feature_matrix = scaler.fit_transform(feature_matrix)
        else:
            feature_matrix = scaler.transform(feature_matrix)

        if fit_encoder:
            classes_encoded = encoder.fit_transform(classes)
        else:
            classes_encoded = encoder.transform(classes)

        sample_meta = {k: np.asarray(v) for k, v in sample_meta.items()}

        return SplitData(
            feature_matrix.astype(np.float32),
            positions.astype(np.float32),
            classes_encoded.astype(np.int64),
            sample_meta,
        )

    @staticmethod
    def _fill_nan(array: np.ndarray) -> np.ndarray:
        if not np.isnan(array).any():
            return array
        col_means = np.nanmean(array, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(np.isnan(array))
        array[inds] = col_means[inds[1]]
        return array

    def _create_anchor_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        sort_cols = [c for c in ["timestamp", "group", "point_id", "ID"] if c in df.columns]
        ordered = df.sort_values(sort_cols) if sort_cols else df

        windows, pos_targets, class_targets = [], [], []
        anchor_ids: list[int] = []
        group_indices: list[int] = []
        group_cols = [c for c in ["timestamp", "group", "point_id", "x_m", "y_m", "nlos_type"] if c in ordered.columns]
        grouped = ordered.groupby(group_cols) if group_cols else [(None, ordered)]

        group_counter = -1
        for _, group in grouped:
            if group.empty:
                continue
            group_counter += 1
            x_coord = group["x_m"].iloc[0] if "x_m" in group.columns else 0.0
            y_coord = group["y_m"].iloc[0] if "y_m" in group.columns else 0.0
            env_label = group["nlos_type"].iloc[0] if "nlos_type" in group.columns else 0
            point_id = group["point_id"].iloc[0] if "point_id" in group.columns else -1

            for _, anchor_row in group.iterrows():
                main_features = [
                    anchor_row["DIST"],
                    anchor_row["RSSI"],
                    anchor_row["SEQ"],
                    anchor_row["LOSS"],
                    anchor_row["CIR"],
                    anchor_row["FP2"],
                    anchor_row["MaxNoise"],
                    anchor_row["SNR"],
                ]

                aux_features: list[float] = []
                for anchor_id in self.anchors:
                    if anchor_id == anchor_row.get("ID"):
                        continue
                    aux = group[group["ID"] == anchor_id]
                    if not aux.empty:
                        ref = aux.iloc[0]
                        aux_features.extend([ref["DIST"], ref["RSSI"], ref["CIR"], ref["SNR"]])
                        continue

                    same_point = df[(df.get("point_id") == point_id) & (df.get("ID") == anchor_id)]
                    if not same_point.empty:
                        aux_features.extend([
                            same_point["DIST"].median(),
                            same_point["RSSI"].median(),
                            same_point["CIR"].median(),
                            same_point["SNR"].median(),
                        ])
                        continue

                    anchor_history = df[df.get("ID") == anchor_id]
                    if not anchor_history.empty:
                        aux_features.extend([
                            anchor_history["DIST"].median(),
                            anchor_history["RSSI"].median(),
                            anchor_history["CIR"].median(),
                            anchor_history["SNR"].median(),
                        ])
                    else:
                        env_slice = df[df.get("nlos_type") == env_label]
                        aux_features.extend([
                            env_slice["DIST"].mean() if not env_slice.empty else 0.0,
                            env_slice["RSSI"].mean() if not env_slice.empty else 0.0,
                            env_slice["CIR"].mean() if not env_slice.empty else 0.0,
                        env_slice["SNR"].mean() if not env_slice.empty else 0.0,
                    ])

                feature_vector = main_features + aux_features
                if len(feature_vector) != 20:
                    raise ValueError(f"Expected 20 features per window, got {len(feature_vector)}")

                windows.append(feature_vector)
                pos_targets.append([x_coord, y_coord])
                class_targets.append(env_label)
                anchor_ids.append(int(anchor_row.get("ID", -1)))
                group_indices.append(group_counter)

        metadata = {
            "anchor_id": np.array(anchor_ids, dtype=np.int32),
            "group_index": np.array(group_indices, dtype=np.int32),
        }

        return (
            np.array(windows, dtype=np.float32),
            np.array(pos_targets, dtype=np.float32),
            np.array(class_targets, dtype=np.int64),
            metadata,
        )

    def _store_cache(
        self,
        train: SplitData,
        val: Optional[SplitData],
        test: SplitData,
        scaler: StandardScaler,
        encoder: LabelEncoder,
        metadata: Dict[str, object],
    ) -> None:
        payload = {
            "train_X": train.X,
            "train_y_pos": train.y_pos,
            "train_y_class": train.y_class,
            "test_X": test.X,
            "test_y_pos": test.y_pos,
            "test_y_class": test.y_class,
        }
        for key, array in train.meta.items():
            payload[f"train_meta_{key}"] = array
        for key, array in test.meta.items():
            payload[f"test_meta_{key}"] = array
        if val:
            payload["val_X"] = val.X
            payload["val_y_pos"] = val.y_pos
            payload["val_y_class"] = val.y_class
            for key, array in val.meta.items():
                payload[f"val_meta_{key}"] = array
        np.savez_compressed(self.cache_path, **payload)

        with open(self.scaler_path, "wb") as fh:
            pickle.dump(scaler, fh)
        with open(self.encoder_path, "wb") as fh:
            pickle.dump(encoder, fh)
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    def _load_cache(self) -> DataBundle:
        npz = np.load(self.cache_path, allow_pickle=True)
        with open(self.scaler_path, "rb") as fh:
            scaler: StandardScaler = pickle.load(fh)
        with open(self.encoder_path, "rb") as fh:
            encoder: LabelEncoder = pickle.load(fh)
        metadata = {}
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)

        def _extract_meta(prefix: str, count: int) -> Dict[str, np.ndarray]:
            collected: Dict[str, np.ndarray] = {}
            token = f"{prefix}_meta_"
            for name in npz.files:
                if name.startswith(token):
                    collected[name[len(token):]] = npz[name]
            if not collected:
                collected = {"anchor_id": np.full((count,), -1, dtype=np.int32), "group_index": np.arange(count, dtype=np.int32)}
            return collected

        train_meta = _extract_meta("train", npz["train_X"].shape[0])
        test_meta = _extract_meta("test", npz["test_X"].shape[0])

        train = SplitData(npz["train_X"], npz["train_y_pos"], npz["train_y_class"], train_meta)
        val = None
        if {"val_X", "val_y_pos", "val_y_class"}.issubset(npz.files):
            val_meta = _extract_meta("val", npz["val_X"].shape[0])
            val = SplitData(npz["val_X"], npz["val_y_pos"], npz["val_y_class"], val_meta)
        test = SplitData(npz["test_X"], npz["test_y_pos"], npz["test_y_class"], test_meta)

        adapters, feature_store = self._build_feature_variants(train, val, test)
        return DataBundle(train, val, test, scaler, encoder, metadata, adapters, feature_store)

    def _build_feature_variants(
        self,
        train: SplitData,
        val: Optional[SplitData],
        test: SplitData,
    ) -> Tuple[InputAdapterManager, Dict[str, Dict[str, np.ndarray]]]:
        manager = build_default_manager()
        if train.X.size:
            manager.fit(train.X)
        store: Dict[str, Dict[str, np.ndarray]] = {}
        for name, split in {"train": train, "val": val, "test": test}.items():
            if split is None:
                store[name] = {}
                continue
            split_store: Dict[str, np.ndarray] = {}
            split_store["base"] = split.X
            profiles = manager.available_profiles()
            for profile, cfg in profiles.items():
                if split.X.size:
                    split_store[profile] = manager.transform(split.X, profile)
                else:
                    split_store[profile] = np.zeros((0, cfg.target_dim), dtype=np.float32)
            store[name] = split_store
        return manager, store
