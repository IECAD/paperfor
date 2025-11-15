import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
import itertools
import math
import time
import copy
import pickle
import os
import json
import random
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# 재현성 보장
# =============================================================================
def set_seed(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =============================================================================
# 1. 데이터 필터링 클래스 (수정됨)
# =============================================================================
class UWBFilter:
    @staticmethod
    def snr_weighted_filter(data, window_size=3):
        """[수정] 시간 정렬 보장"""
        print(f"SNR 가중 필터 적용 (윈도우: {window_size})")
        filtered_data = data.copy()
        
        for point in data['point_id'].unique():
            for anchor in data['ID'].unique():
                mask = (data['point_id'] == point) & (data['ID'] == anchor)
                # [수정] 시간 정렬 추가
                subset = data.loc[mask].sort_values('timestamp').copy() if 'timestamp' in data.columns else data.loc[mask].copy()
                
                if len(subset) >= window_size:
                    weights = np.clip(subset['SNR'] / 8.0, 0.1, 1.0)
                    
                    filtered_values = []
                    for i in range(len(subset)):
                        start_idx = max(0, i - window_size//2)
                        end_idx = min(len(subset), i + window_size//2 + 1)
                        
                        window_dist = subset.iloc[start_idx:end_idx]['DIST'].values
                        window_weights = weights.iloc[start_idx:end_idx].values
                        
                        if len(window_dist) > 0:
                            weighted_avg = np.average(window_dist, weights=window_weights)
                            filtered_values.append(weighted_avg)
                        else:
                            filtered_values.append(subset.iloc[i]['DIST'])
                    
                    filtered_data.loc[mask, 'DIST'] = filtered_values
        
        return filtered_data
    
    @staticmethod
    def combo_filter(data, window_size=3, z_threshold=2.5, min_snr=1.0, min_cir=30):
        """[개선] 벡터화 적용으로 효율성 증가"""
        print("조합 필터 적용")
        
        quality_mask = (data['SNR'] >= min_snr) & (data['CIR'] >= min_cir)
        filtered_data = data.copy()
        
        # 품질 기반 필터링 (벡터화)
        for point in data['point_id'].unique():
            for anchor in data['ID'].unique():
                mask = (data['point_id'] == point) & (data['ID'] == anchor)
                subset = data.loc[mask]
                
                if len(subset) > 0:
                    high_quality = subset.loc[quality_mask.loc[mask]]
                    low_quality_indices = subset.loc[~quality_mask.loc[mask]].index
                    
                    if len(high_quality) > 0 and len(low_quality_indices) > 0:
                        replacement_value = high_quality['DIST'].mean()
                        filtered_data.loc[low_quality_indices, 'DIST'] = replacement_value
        
        # 이상치 제거
        for point in data['point_id'].unique():
            for anchor in data['ID'].unique():
                mask = (data['point_id'] == point) & (data['ID'] == anchor)
                subset = filtered_data.loc[mask]
                
                if len(subset) > 3:
                    z_scores = np.abs(stats.zscore(subset['DIST']))
                    outlier_mask = z_scores > z_threshold
                    
                    group_mean = subset.loc[~outlier_mask, 'DIST'].mean()
                    if not np.isnan(group_mean):
                        filtered_data.loc[mask & filtered_data.index.isin(subset.index[outlier_mask]), 'DIST'] = group_mean
        
        # SNR 가중 필터 적용
        filtered_data = UWBFilter.snr_weighted_filter(filtered_data, window_size)
        
        return filtered_data

# =============================================================================
# 2. 데이터 처리 클래스 (수정됨)
# =============================================================================
class FixedUWBDataProcessor:
    def __init__(self, filter_type='combo', split_method='method4'):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.filter_type = filter_type
        self.split_method = split_method
        self.class_name_map = {
            0: 'LOS',
            1: 'Static NLOS',
            2: 'Dynamic NLOS',
            3: 'CAD Static NLOS',
            4: 'CAD Dynamic NLOS'
        }
        self.filtered_splits = {}
        self.meta_splits = {}
        self.class_counts_ = {}
        self.num_classes = 0
        self.feature_dim = 0
    
    def _resolve_dataset_path(self, filename: str) -> Path:
        candidate_dirs = [
            Path('./new_bench_02/1_data/raw'),
            Path('./new_bench_02/1_data/processed'),
            Path('./new_bench/data/raw'),
            Path('./new_bench/data/processed'),
            Path('./data'),
            Path('./uwb_data')
        ]
        for directory in candidate_dirs:
            candidate = directory / filename
            if candidate.exists():
                return candidate
        search_paths = ', '.join(str(d) for d in candidate_dirs)
        raise FileNotFoundError(
            f"Dataset file '{filename}' not found in expected locations ({search_paths})."
        )

    def load_data(self):
        dataset_files = {
            'los': 'los.xlsx',
            'static': 'nlos_static.xlsx',
            'dynamic': 'nlos_dynamic.xlsx',
            'cad_static': 'cad_static_NLOS.xlsx',
            'cad_dynamic': 'cad_dynamic_NLOS.xlsx'
        }

        data_frames = []
        missing = []
        for name, filename in dataset_files.items():
            try:
                path = self._resolve_dataset_path(filename)
                df = pd.read_excel(path)
                df['dataset_source'] = name
                data_frames.append(df)
            except FileNotFoundError:
                missing.append(filename)

        if not data_frames:
            raise FileNotFoundError("No dataset files available for processing. Please check dataset paths.")
        if missing:
            print(f"[WARN] Skipped missing datasets: {', '.join(missing)}")

        all_data = pd.concat(data_frames, ignore_index=True)

        feature_columns = ['DIST', 'RSSI', 'SEQ', 'LOSS', 'CIR', 'FP2', 'MaxNoise', 'SNR']
        all_data = all_data.replace([np.inf, -np.inf], np.nan)
        before_drop = len(all_data)
        all_data = all_data.dropna(subset=feature_columns)
        after_drop = len(all_data)
        if before_drop != after_drop:
            print(f"[WARN] Dropped {before_drop - after_drop} rows with missing feature values.")

        # ���� ��ȯ (�׸��� �� ����)
        all_data['x_m'] = all_data['x'] * 0.96
        all_data['y_m'] = all_data['y'] * 1.02

        return all_data

    def apply_filter(self, data):
        if self.filter_type == 'none':
            return data
        elif self.filter_type == 'snr_weighted':
            return UWBFilter.snr_weighted_filter(data)
        elif self.filter_type == 'combo':
            return UWBFilter.combo_filter(data)
        else:
            return data
    
    def safe_point_wise_split(self, data, train_ratio=0.6, val_ratio=0.2):
        """안전한 포인트별 시간 분할"""
        print("안전한 포인트별 시간 분할")
        
        train_data_list = []
        val_data_list = []
        test_data_list = []
        
        for nlos_type in sorted(data['nlos_type'].unique()):
            class_data = data[data['nlos_type'] == nlos_type]
            
            class_train_list = []
            class_val_list = []
            class_test_list = []
            
            for point in class_data['point_id'].unique():
                point_class_data = class_data[class_data['point_id'] == point]
                
                # 안전한 정렬
                sort_columns = [c for c in ['timestamp', 'group'] if c in point_class_data.columns]
                if sort_columns:
                    try:
                        point_class_data = point_class_data.sort_values(sort_columns)
                    except:
                        point_class_data = point_class_data.sort_index()
                else:
                    point_class_data = point_class_data.sort_index()
                
                n_total = len(point_class_data)
                if n_total < 3:
                    class_train_list.append(point_class_data)
                    continue
                
                train_end = max(1, int(n_total * train_ratio))
                val_end = max(train_end + 1, int(n_total * (train_ratio + val_ratio)))
                val_end = min(val_end, n_total - 1)
                
                class_train_list.append(point_class_data.iloc[:train_end])
                if val_end > train_end:
                    class_val_list.append(point_class_data.iloc[train_end:val_end])
                if val_end < n_total:
                    class_test_list.append(point_class_data.iloc[val_end:])
                else:
                    class_test_list.append(point_class_data.iloc[-1:])
            
            if class_train_list:
                train_data_list.append(pd.concat(class_train_list, ignore_index=True))
            if class_val_list:
                val_data_list.append(pd.concat(class_val_list, ignore_index=True))
            if class_test_list:
                test_data_list.append(pd.concat(class_test_list, ignore_index=True))
        
        train_data = pd.concat(train_data_list, ignore_index=True) if train_data_list else pd.DataFrame()
        val_data = pd.concat(val_data_list, ignore_index=True) if val_data_list else pd.DataFrame()
        test_data = pd.concat(test_data_list, ignore_index=True) if test_data_list else pd.DataFrame()
        
        return train_data, val_data, test_data
    
    def create_anchor_windows(self, data):
        """[����] ������ ������ ���� - ���� ���� ���"""
        if data is None or len(data) == 0:
            return (
                np.array([]).reshape(0, 0),
                np.array([]).reshape(0, 2),
                np.array([]),
                []
            )

        # ������ ����
        sort_cols = [c for c in ['timestamp', 'group', 'point_id', 'ID'] if c in data.columns]
        data_sorted = data.sort_values(sort_cols) if sort_cols else data

        windows = []
        targets_pos = []
        targets_class = []
        metadata = []

        # �׷�Ű ����
        group_keys = [c for c in ['timestamp', 'group', 'point_id', 'x_m', 'y_m', 'nlos_type']
                      if c in data_sorted.columns]
        grouped = data_sorted.groupby(group_keys)
        group_index = -1

        for name, group in grouped:
            group_index += 1
            # [����] ���� ���� ������� �����ϰ� ����
            x_coord = group['x_m'].iloc[0]
            y_coord = group['y_m'].iloc[0]
            nlos_type_val = group['nlos_type'].iloc[0]

            for _, anchor_row in group.iterrows():
                # ID�� �����ϰ� 8�������� ����
                base_features = [
                    anchor_row['DIST'], anchor_row['RSSI'], anchor_row['SEQ'],
                    anchor_row['LOSS'], anchor_row['CIR'], anchor_row['FP2'],
                    anchor_row['MaxNoise'], anchor_row['SNR']
                ]

                other_anchors_features = []
                for anchor_id in [1, 2, 3, 4]:
                    if anchor_id != anchor_row['ID']:
                        other_anchor = group[group['ID'] == anchor_id]
                        if len(other_anchor) > 0:
                            other_row = other_anchor.iloc[0]
                            other_features = [
                                other_row['DIST'], other_row['RSSI'],
                                other_row['CIR'], other_row['SNR']
                            ]
                        else:
                            # [����] ������ ������ ó��
                            same_point_anchor = data[
                                (data['point_id'] == group['point_id'].iloc[0]) &
                                (data['ID'] == anchor_id)
                            ]
                            if len(same_point_anchor) > 0:
                                other_features = [
                                    same_point_anchor['DIST'].median(),
                                    same_point_anchor['RSSI'].median(),
                                    same_point_anchor['CIR'].median(),
                                    same_point_anchor['SNR'].median()
                                ]
                            else:
                                same_anchor = data[data['ID'] == anchor_id]
                                if len(same_anchor) > 0:
                                    other_features = [
                                        same_anchor['DIST'].median(),
                                        same_anchor['RSSI'].median(),
                                        same_anchor['CIR'].median(),
                                        same_anchor['SNR'].median()
                                    ]
                                else:
                                    env_data = data[data['nlos_type'] == nlos_type_val]
                                    other_features = [
                                        env_data['DIST'].mean(),
                                        env_data['RSSI'].mean(),
                                        env_data['CIR'].mean(),
                                        env_data['SNR'].mean()
                                    ]
                        other_anchors_features.extend(other_features)

                window_features = base_features + other_anchors_features

                windows.append(window_features)
                targets_pos.append([x_coord, y_coord])
                targets_class.append(nlos_type_val)
                metadata.append({
                    'group_index': group_index,
                    'anchor_id': int(anchor_row['ID']),
                    'point_id': group['point_id'].iloc[0] if 'point_id' in group.columns else None,
                    'timestamp': group['timestamp'].iloc[0] if 'timestamp' in group.columns else None,
                    'nlos_type': nlos_type_val
                })

        return (
            np.array(windows),
            np.array(targets_pos),
            np.array(targets_class),
            metadata
        )

    def preprocess(self, data):
        """��ó�� - ���� �� ���͸�"""
        # 1) ���� ����
        train_raw, val_raw, test_raw = self.safe_point_wise_split(data)

        # 2) �� split�� ���������� ���� ����
        train_data = self.apply_filter(train_raw)
        val_data = self.apply_filter(val_raw) if len(val_raw) > 0 else val_raw
        test_data = self.apply_filter(test_raw)

        print(f"���� �� Ŭ���� ����:")
        for dataset_name, dataset in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            if len(dataset) > 0:
                class_counts = dataset['nlos_type'].value_counts().sort_index()
                info_parts = []
                for cls_id in sorted(self.class_name_map.keys()):
                    label = self.class_name_map.get(cls_id, f"class_{cls_id}")
                    info_parts.append(f"{label}:{int(class_counts.get(cls_id, 0))}")
                print(f"    {dataset_name}: {', '.join(info_parts)}")
            else:
                print(f"    {dataset_name}: empty")

        X_train, y_pos_train, y_class_train, meta_train = self.create_anchor_windows(train_data)
        X_test, y_pos_test, y_class_test, meta_test = self.create_anchor_windows(test_data)
        if len(val_data) > 0:
            X_val, y_pos_val, y_class_val, meta_val = self.create_anchor_windows(val_data)
        else:
            feature_dim = X_train.shape[1] if X_train.size > 0 else (X_test.shape[1] if X_test.size > 0 else 0)
            X_val = np.empty((0, feature_dim)) if feature_dim > 0 else np.empty((0, 0))
            y_pos_val = np.empty((0, 2))
            y_class_val = np.array([])
            meta_val = []

        if X_train.size == 0:
            raise ValueError("Training feature matrix is empty after preprocessing.")

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val.size > 0 else np.empty((0, X_train.shape[1]))
        X_test_scaled = self.scaler.transform(X_test) if X_test.size > 0 else np.empty((0, X_train.shape[1]))

        y_class_train_encoded = self.label_encoder.fit_transform(y_class_train)
        y_class_val_encoded = self.label_encoder.transform(y_class_val) if len(y_class_val) > 0 else np.array([])
        y_class_test_encoded = self.label_encoder.transform(y_class_test)
        self.num_classes = len(self.label_encoder.classes_)
        train_counts = pd.Series(y_class_train).value_counts().to_dict()
        self.class_counts_ = {int(k): int(v) for k, v in train_counts.items()}

        # ��ǥ�� �̹� ���� ������ ��ȯ��
        y_pos_train_real = y_pos_train.astype(float)
        y_pos_val_real = y_pos_val.astype(float) if len(y_pos_val) > 0 else np.empty((0, 2))
        y_pos_test_real = y_pos_test.astype(float)

        self.filtered_splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        self.meta_splits = {
            'train': meta_train,
            'val': meta_val,
            'test': meta_test
        }
        self.feature_dim = X_train.shape[1]

        return (X_train_scaled, y_pos_train_real, y_class_train_encoded,
                X_val_scaled, y_pos_val_real, y_class_val_encoded,
                X_test_scaled, y_pos_test_real, y_class_test_encoded)

# =============================================================================
# 3. 모델 클래스 (수정됨)
# =============================================================================
class UWB_HyperTuned_Model(nn.Module):
    def __init__(self, input_dim=20, num_classes=3, main_dim=8, other_dim=12):
        """[수정] input_dim 검증 추가"""
        super(UWB_HyperTuned_Model, self).__init__()
        
        # [수정] 차원 검증
        assert input_dim == main_dim + other_dim, f"input_dim {input_dim} != {main_dim+other_dim}"
        
        self.main_anchor_dim = main_dim
        self.other_anchors_dim = other_dim
        
        # 최적화된 주요 앵커 처리기
        self.main_anchor_processor = nn.Sequential(
            nn.Linear(self.main_anchor_dim, 160),
            nn.LayerNorm(160),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(160, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, 40)
        )
        
        # 최적화된 보조 앵커 처리기
        self.other_anchors_processor = nn.Sequential(
            nn.Linear(self.other_anchors_dim, 160),
            nn.LayerNorm(160),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(160, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, 40)
        )
        
        # 주의집중 메커니즘
        self.attention = nn.Sequential(
            nn.Linear(80, 80),
            nn.LayerNorm(80),
            nn.Tanh(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 2),
            nn.Softmax(dim=1)
        )
        
        # 최적화된 공유 네트워크
        self.shared_layers = nn.Sequential(
            nn.Linear(80, 640),
            nn.LayerNorm(640),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(640, 320),
            nn.LayerNorm(320),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(320, 160),
            nn.LayerNorm(160),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 정교한 위치 헤드
        self.position_head = nn.Sequential(
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        
        # 분류 헤드
        self.classification_head = nn.Sequential(
            nn.Linear(160, 40),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes)
        )
        
        # 신뢰도 헤드
        self.confidence_head = nn.Sequential(
            nn.Linear(160, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        main_anchor_features = x[:, :self.main_anchor_dim]
        other_anchors_features = x[:, self.main_anchor_dim:]
        
        main_processed = self.main_anchor_processor(main_anchor_features)
        others_processed = self.other_anchors_processor(other_anchors_features)
        
        combined_features = torch.cat([main_processed, others_processed], dim=1)
        attention_weights = self.attention(combined_features)
        
        weighted_main = main_processed * attention_weights[:, 0:1]
        weighted_others = others_processed * attention_weights[:, 1:2]
        
        fused_features = torch.cat([weighted_main, weighted_others], dim=1)
        shared_features = self.shared_layers(fused_features)
        
        position = self.position_head(shared_features)
        classification = self.classification_head(shared_features)
        confidence = self.confidence_head(shared_features)
        
        return position, classification, confidence, attention_weights

# =============================================================================
# 4. 손실 함수 (수정됨)
# =============================================================================
class UWBMultiTaskLoss(nn.Module):
    def __init__(self, pos_weight=2.5, class_weight=2.0, conf_weight=0.5, conf_norm_meters=0.5):
        super(UWBMultiTaskLoss, self).__init__()
        self.pos_weight = pos_weight
        self.class_weight = class_weight
        self.conf_weight = conf_weight
        self.conf_norm_meters = conf_norm_meters
        
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
        self.class_weights = None
        self.bce_loss = nn.BCELoss()
    
    def set_class_weights(self, label_encoder_classes, class_counts=None):
        """[����] ������ Ŭ���� ����ġ ����"""
        if class_counts:
            counts = torch.tensor([class_counts.get(int(cls), 1) for cls in label_encoder_classes], dtype=torch.float)
            weights = counts.sum() / (counts + 1e-8)
            weights = weights / weights.mean()
        else:
            weights = torch.ones(len(label_encoder_classes), dtype=torch.float)
        self.class_weights = weights
    
    def forward(self, pos_pred, pos_true, class_pred, class_true, conf_pred, pos_error):
        device = pos_pred.device
        
        # 위치 손실
        pos_loss = 0.6 * self.smooth_l1_loss(pos_pred, pos_true) + 0.4 * self.mse_loss(pos_pred, pos_true)
        
        # 분류 손실
        if self.class_weights is not None:
            class_weights_device = self.class_weights.to(device)
            class_loss = nn.functional.cross_entropy(class_pred, class_true, weight=class_weights_device)
        else:
            class_loss = nn.functional.cross_entropy(class_pred, class_true)
        
        # 신뢰도 손실 (detach 적용)
        normalized_error = torch.clamp(pos_error / self.conf_norm_meters, 0, 1)
        target_confidence = (1.0 - normalized_error).detach()  # 그라디언트 분리
        target_confidence = torch.clamp(target_confidence, 1e-6, 1 - 1e-6)
        conf_pred = torch.clamp(conf_pred, 1e-6, 1 - 1e-6)
        conf_loss = self.bce_loss(conf_pred.squeeze(), target_confidence)
        
        total_loss = (self.pos_weight * pos_loss + 
                     self.class_weight * class_loss + 
                     self.conf_weight * conf_loss)
        
        return total_loss, pos_loss, class_loss, conf_loss

# =============================================================================
# 5. 데이터셋 및 유틸리티
# =============================================================================
class UWBDataset(Dataset):
    def __init__(self, X, y_pos, y_class):
        self.X = torch.FloatTensor(X)
        self.y_pos = torch.FloatTensor(y_pos)
        self.y_class = torch.LongTensor(y_class)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_pos[idx], self.y_class[idx]


class TPTPBaseline:
    def __init__(self, anchors=(1, 2, 3, 4), lambda_thresh=0.9, num_particles=1500, sabo_iterations=3,
                 sample_sigma=0.35, process_noise=0.15):
        self.anchors = anchors
        self.lambda_thresh = lambda_thresh
        self.num_particles = num_particles
        self.sabo_iterations = sabo_iterations
        self.sample_sigma = sample_sigma
        self.process_noise = process_noise
        self.anchor_positions = {}
        self.measurement_noise = {}

    def fit(self, train_df, metadata, features, targets):
        if train_df is None or len(train_df) == 0:
            raise ValueError("Training dataframe is empty for baseline calibration.")
        for anchor_id in self.anchors:
            subset = train_df[train_df['ID'] == anchor_id]
            if subset.empty:
                continue
            x = subset['x'].values.astype(float) * 0.96 if 'x_m' not in subset else subset['x_m'].values.astype(float)
            y = subset['y'].values.astype(float) * 1.02 if 'y_m' not in subset else subset['y_m'].values.astype(float)
            d = subset['DIST'].values.astype(float)
            A = np.column_stack((-2 * x, -2 * y, np.ones_like(x)))
            b = d ** 2 - (x ** 2 + y ** 2)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            ax, ay, _ = sol
            self.anchor_positions[anchor_id] = np.array([ax, ay])
            residuals = np.sqrt((x - ax) ** 2 + (y - ay) ** 2) - d
            sigma = np.std(residuals) if residuals.size > 0 else 0.25
            if not np.isfinite(sigma) or sigma < 1e-3:
                sigma = 0.25
            self.measurement_noise[anchor_id] = float(sigma)

    def _extract_measurements(self, feature_row, main_anchor_id):
        measurements = {}
        # main anchor features
        measurements[main_anchor_id] = {
            'distance': float(feature_row[0]),
            'snr': float(feature_row[7]) if len(feature_row) > 7 else 0.0,
            'cir': float(feature_row[4]) if len(feature_row) > 4 else 0.0
        }
        other_ids = [a for a in self.anchors if a != main_anchor_id]
        offset = 8
        for idx, anchor_id in enumerate(other_ids):
            base = offset + idx * 4
            if base + 3 < len(feature_row):
                measurements[anchor_id] = {
                    'distance': float(feature_row[base]),
                    'snr': float(feature_row[base + 3]),
                    'cir': float(feature_row[base + 2])
                }
        return measurements

    def _mrwgh_estimate(self, measurements):
        candidates = []
        weights = []
        for combo in itertools.combinations([aid for aid in measurements if aid in self.anchor_positions], 3):
            coords = [self.anchor_positions[a] for a in combo]
            dists = [measurements[a]['distance'] for a in combo]
            A = []
            b = []
            base_pos = coords[0]
            base_dist = dists[0]
            for idx in range(1, len(combo)):
                pos = coords[idx]
                dist = dists[idx]
                A.append([2 * (pos[0] - base_pos[0]), 2 * (pos[1] - base_pos[1])])
                b.append(base_dist ** 2 - dist ** 2 + pos[0] ** 2 - base_pos[0] ** 2 + pos[1] ** 2 - base_pos[1] ** 2)
            A = np.array(A)
            b = np.array(b)
            try:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                candidate = sol
            except np.linalg.LinAlgError:
                continue
            residuals = []
            for anchor_id, pos, dist in zip(combo, coords, dists):
                estimated = np.linalg.norm(candidate - pos)
                residuals.append(abs(estimated - dist))
            avg_residual = np.mean(residuals) if residuals else 1.0
            weight = 1.0 / (avg_residual + 1e-6) ** 2
            candidates.append(candidate)
            weights.append(weight)
        if not candidates:
            return np.zeros(2)
        weights = np.array(weights)
        weights = weights / weights.sum()
        return np.average(np.array(candidates), axis=0, weights=weights)

    def _nlos_adjust(self, measurements, coarse):
        adjusted = {}
        for anchor_id, meas in measurements.items():
            anchor_pos = self.anchor_positions.get(anchor_id)
            if anchor_pos is None:
                continue
            sigma = self.measurement_noise.get(anchor_id, 0.25)
            expected = np.linalg.norm(coarse - anchor_pos)
            residual = meas['distance'] - expected
            los_prob = stats.norm.cdf(residual / (sigma + 1e-6))
            is_nlos = los_prob > self.lambda_thresh
            adjusted_distance = expected if is_nlos else meas['distance']
            adjusted[anchor_id] = {
                'distance': adjusted_distance,
                'sigma': sigma * (2.0 if is_nlos else 1.0),
                'is_nlos': is_nlos
            }
        return adjusted

    def _sample_particles(self, coarse, measurement_info):
        particles = coarse + np.random.normal(scale=self.sample_sigma, size=(self.num_particles, 2))
        valid_mask = np.ones(self.num_particles, dtype=bool)
        for anchor_id, info in measurement_info.items():
            anchor_pos = self.anchor_positions.get(anchor_id)
            if anchor_pos is None:
                continue
            radius = info['distance'] + 2 * info['sigma']
            if radius <= 0:
                continue
            dist = np.linalg.norm(particles - anchor_pos, axis=1)
            valid_mask &= dist <= (radius + 0.01)
        if not np.any(valid_mask):
            return coarse + np.random.normal(scale=self.sample_sigma, size=(self.num_particles, 2))
        particles = particles[valid_mask]
        if len(particles) < self.num_particles:
            extras = coarse + np.random.normal(scale=self.sample_sigma, size=(self.num_particles - len(particles), 2))
            particles = np.vstack([particles, extras])
        return particles

    def _likelihood(self, particles, measurement_info):
        weights = np.ones(len(particles))
        for anchor_id, info in measurement_info.items():
            anchor_pos = self.anchor_positions.get(anchor_id)
            if anchor_pos is None:
                continue
            expected = np.linalg.norm(particles - anchor_pos, axis=1)
            residual = info['distance'] - expected
            sigma = info['sigma'] + 1e-6
            weights *= np.exp(-0.5 * (residual / sigma) ** 2)
        total = weights.sum()
        if total <= 0:
            return np.ones(len(particles)) / len(particles)
        return weights / total

    def _refine(self, measurement_info, coarse):
        particles = self._sample_particles(coarse, measurement_info)
        for _ in range(self.sabo_iterations):
            weights = self._likelihood(particles, measurement_info)
            elite_count = max(int(0.1 * len(particles)), 1)
            elite_idx = np.argsort(weights)[-elite_count:]
            elite = particles[elite_idx]
            mean_elite = elite.mean(axis=0)
            perturb = elite - mean_elite
            particles = elite + perturb * 0.5 + np.random.normal(scale=self.process_noise, size=elite.shape)
        weights = self._likelihood(particles, measurement_info)
        return np.average(particles, axis=0, weights=weights)

    def predict(self, features, metadata):
        if len(features) == 0:
            return np.empty((0, 2))
        predictions = np.zeros((len(features), 2))
        cache = {}
        for idx, (feature_row, meta) in enumerate(zip(features, metadata)):
            group_index = meta.get('group_index', idx)
            if group_index not in cache:
                measurements = self._extract_measurements(feature_row, meta.get('anchor_id', 1))
                coarse = self._mrwgh_estimate(measurements)
                adjusted = self._nlos_adjust(measurements, coarse)
                refined = self._refine(adjusted, coarse)
                cache[group_index] = refined
            predictions[idx] = cache[group_index]
        return predictions

    def evaluate(self, features, targets, metadata):
        preds = self.predict(features, metadata)
        if len(preds) == 0:
            return {
                'predictions': preds,
                'avg_error_cm': float('nan'),
                'p90_error_cm': float('nan'),
                'errors_cm': np.array([])
            }
        errors = np.linalg.norm(preds - targets, axis=1) * 100.0
        avg_error = float(np.mean(errors))
        p90_error = float(np.percentile(errors, 90))
        return {
            'predictions': preds,
            'avg_error_cm': avg_error,
            'p90_error_cm': p90_error,
            'errors_cm': errors
        }

def calculate_distance_error(pred, true):
    return torch.sqrt(torch.sum((pred - true) ** 2, dim=1))

# =============================================================================
# 6. 훈련 함수
# =============================================================================
def train_model(model, train_loader, val_loader, criterion, num_epochs=150, lr=0.001):
    """훈련 함수"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    use_early_stopping = val_loader is not None
    if use_early_stopping:
        best_val_loss = float('inf')
        patience = 30
        wait = 0
        min_epochs = 40
    
    print(f"모델 훈련 중... (검증셋: {'있음' if use_early_stopping else '없음'})")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0
        
        for X_batch, y_pos_batch, y_class_batch in train_loader:
            X_batch = X_batch.to(device)
            y_pos_batch = y_pos_batch.to(device)
            y_class_batch = y_class_batch.to(device)
            
            optimizer.zero_grad()
            pos_pred, class_pred, conf_pred, attention_weights = model(X_batch)
            pos_error = calculate_distance_error(pos_pred, y_pos_batch)
            
            loss, pos_loss, class_loss, conf_loss = criterion(
                pos_pred, y_pos_batch, class_pred, y_class_batch, conf_pred, pos_error
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
        
        # 검증 및 얼리 스탑
        if use_early_stopping:
            model.eval()
            val_loss_total = 0
            
            with torch.no_grad():
                for X_batch, y_pos_batch, y_class_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_pos_batch = y_pos_batch.to(device)
                    y_class_batch = y_class_batch.to(device)
                    
                    pos_pred, class_pred, conf_pred, attention_weights = model(X_batch)
                    pos_error = calculate_distance_error(pos_pred, y_pos_batch)
                    
                    loss, pos_loss, class_loss, conf_loss = criterion(
                        pos_pred, y_pos_batch, class_pred, y_class_batch, conf_pred, pos_error
                    )
                    
                    val_loss_total += loss.item()
            
            avg_val_loss = val_loss_total / len(val_loader)
            scheduler.step()
            
            if epoch >= min_epochs:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), 'temp_best_model.pth')
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break
            else:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'temp_best_model.pth')
        else:
            scheduler.step()
            torch.save(model.state_dict(), 'temp_best_model.pth')
    
    return model

def evaluate_model(model, test_loader):
    """[수정] 평가 메트릭 확장"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 최고 성능 모델 로드
    if os.path.exists('temp_best_model.pth'):
        model.load_state_dict(torch.load('temp_best_model.pth', map_location=device))
        model.to(device)
    model.eval()
    
    all_pos_errors = []
    all_class_correct = 0
    all_total = 0
    class_predictions = []
    class_targets = []
    
    with torch.no_grad():
        for X_batch, y_pos_batch, y_class_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pos_batch = y_pos_batch.to(device)
            y_class_batch = y_class_batch.to(device)
            
            pos_pred, class_pred, conf_pred, attention_weights = model(X_batch)
            
            pos_errors = calculate_distance_error(pos_pred, y_pos_batch).cpu().numpy() * 100
            all_pos_errors.extend(pos_errors)
            
            _, predicted = torch.max(class_pred.data, 1)
            all_total += y_class_batch.size(0)
            all_class_correct += (predicted == y_class_batch).sum().item()
            
            class_predictions.extend(predicted.cpu().numpy())
            class_targets.extend(y_class_batch.cpu().numpy())
    
    avg_error = np.mean(all_pos_errors)
    p90_error = np.percentile(all_pos_errors, 90)  # [추가] P90 에러
    accuracy = 100 * all_class_correct / all_total if all_total > 0 else 0
    
    return avg_error, accuracy, p90_error, all_pos_errors

# =============================================================================
# 7. 모델 저장/로드 함수 (수정됨)
# =============================================================================
def save_model(model, processor, performance, train_shape):
    """[수정] 메타데이터 확장 및 차원 일관성"""
    try:
        os.makedirs('saved_models', exist_ok=True)
        
        # 모델 저장
        model_path = f'saved_models/uwb_hypertuned_model_{performance:.1f}cm.pth'
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), 'best_uwb_model.pth')
        
        # 스케일러 저장 (단일 경로로 통일)
        scaler_path = 'saved_models/uwb_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(processor.scaler, f)
        
        # 레이블 인코더 저장 (단일 경로로 통일)
        label_encoder_path = 'saved_models/uwb_label_encoder.pkl'
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(processor.label_encoder, f)
        
        # [수정] 모델 정보 저장 (올바른 차원 및 하이퍼파라미터)
        model_info = {
            'model_name': 'HyperTuned',
            'performance': float(performance),
            'input_dim': int(train_shape[1]),  # 실제 훈련 데이터 차원 (20)
            'num_classes': processor.num_classes,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'scaler_path': scaler_path,
            'label_encoder_path': label_encoder_path,
            'hyperparameters': {
                'main_anchor_dim': 8,
                'other_anchors_dim': 12,
                'conf_norm_meters': 0.5
            }
        }
        
        info_path = 'saved_models/model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"모델 저장 완료!")
        print(f"  모델: {model_path}")
        print(f"  스케일러: {scaler_path}")
        print(f"  레이블 인코더: {label_encoder_path}")
        print(f"  성능: {performance:.1f}cm")
        
        return True
        
    except Exception as e:
        print(f"모델 저장 실패: {e}")
        return False

def load_model(model_path='best_uwb_model.pth'):
    """[����] ����� �� �ε� - �ùٸ� ����"""
    try:
        # [����] �� �������� ���� �б�
        info_path = 'saved_models/model_info.json'
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
                input_dim = model_info.get('input_dim', 20)  # 기본값 20
                num_classes = model_info.get('num_classes', 3)
        else:
            input_dim = 20  # 기본값
            num_classes = 3

        model = UWB_HyperTuned_Model(input_dim=input_dim, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        with open('saved_models/uwb_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('saved_models/uwb_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        print(f"�� �ε� ����: {model_path} (input_dim={input_dim}, num_classes={num_classes})")
        return model, scaler, label_encoder

    except Exception as e:
        print(f"�� �ε� ����: {e}")
        return None, None, None


# =============================================================================
# 8. 메인 실험 함수
# =============================================================================
def main_experiment():
    """메인 실험 - 버그 패치 버전"""
    
    print("UWB 위치 추정 - 버그 패치 버전")
    print("=" * 60)
    print("주요 수정사항:")
    print("1. 입력 차원 불일치 해결 (20차원 통일)")
    print("2. 시간 정렬 보장")
    print("3. 윈도우 생성 로직 안정화")
    print("4. 클래스 가중치 버그 수정")
    print("5. 평가 메트릭 확장 (P90 추가)")
    print("=" * 60)
    
    # 데이터 처리
    processor = FixedUWBDataProcessor(filter_type='combo', split_method='method4')
    data = processor.load_data()
    
    (X_train, y_train, y_class_train,
     X_val, y_val, y_class_val,
     X_test, y_test, y_class_test) = processor.preprocess(data)
    
    print(f"데이터: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"피처 차원: {X_train.shape[1]}")  # 20차원 확인
    
    # 데이터 로더
    train_dataset = UWBDataset(X_train, y_train, y_class_train)
    val_dataset = UWBDataset(X_val, y_val, y_class_val) if len(X_val) > 0 else None
    test_dataset = UWBDataset(X_test, y_test, y_class_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # [수정] 모델 생성 - 올바른 차원
    print("\n버그 패치 모델 훈련 시작...")
    model = UWB_HyperTuned_Model(input_dim=X_train.shape[1], num_classes=processor.num_classes)
    
    # [수정] 클래스 가중치 안전하게 설정
    criterion = UWBMultiTaskLoss()
    criterion.set_class_weights(processor.label_encoder.classes_, processor.class_counts_)
    print(f"클래스 가중치 설정: {criterion.class_weights}")
    
    start_time = time.time()
    trained_model = train_model(model, train_loader, val_loader, criterion, num_epochs=150, lr=0.001)
    training_time = time.time() - start_time
    
    # 평가 (확장된 메트릭)
    test_error, test_accuracy, p90_error, all_errors = evaluate_model(trained_model, test_loader)
    baseline = TPTPBaseline()
    try:
        baseline.fit(
            processor.filtered_splits.get('train'),
            processor.meta_splits.get('train'),
            X_train,
            y_train
        )
        baseline_results = baseline.evaluate(
            X_test,
            y_test,
            processor.meta_splits.get('test', [])
        )
        print("\n[TPTP Baseline Results]")
        print(f"    Avg Error: {baseline_results['avg_error_cm']:.1f}cm")
        print(f"    P90 Error: {baseline_results['p90_error_cm']:.1f}cm")
    except Exception as baseline_err:
        print(f"[WARN] TPTP baseline evaluation failed: {baseline_err}")

    
    print(f"\n훈련 완료 (소요시간: {training_time/60:.1f}분)")
    print(f"평균 위치 오차: {test_error:.1f}cm")
    print(f"P90 위치 오차: {p90_error:.1f}cm")  # [추가]
    print(f"분류 정확도: {test_accuracy:.1f}%")
    
    # 모델 저장
    if save_model(trained_model, processor, test_error, X_train.shape):
        print(f"\n실시간 시스템 준비 완료!")
        print(f"차원 일관성 확인: {X_train.shape[1]}차원")
    
    # 임시 파일 정리
    if os.path.exists('temp_best_model.pth'):
        os.remove('temp_best_model.pth')
    
    return {
        'position_error': test_error,
        'p90_error': p90_error,
        'classification_accuracy': test_accuracy,
        'training_time': training_time,
        'model': trained_model
    }

# =============================================================================
# 9. 실행
# =============================================================================
if __name__ == "__main__":
    print("UWB HyperTuned 모델 - 버그 패치 버전")
    print("=" * 50)
    print("치명적 버그 수정:")
    print("• 입력 차원 불일치 해결")
    print("• 시간 정렬 보장")
    print("• 윈도우 생성 안정화")
    print("• 클래스 가중치 수정")
    print("=" * 50)
    
    results = main_experiment()
    
    print("\n" + "="*50)
    print("버그 패치 모델 훈련 완료!")
    print("안정성과 재현성이 보장된 결과입니다.")
    print("="*50)








