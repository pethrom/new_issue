import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, data, indices=None):
        self.data = []

        if indices is not None:
            for i in indices:
                if i + 6 < len(data):
                    seq = data[i:i + 6, 1:6]  # 5个特征列（油量+4个特征）
                    target = data[i + 6, 0]  # 目标值（效率%）
                    if not np.isnan(seq).any() and not np.isnan(target):
                        self.data.append((seq, target))
        else:
            for i in range(len(data) - 6):
                seq = data[i:i + 6, 1:6]
                target = data[i + 6, 0]
                if not np.isnan(seq).any() and not np.isnan(target):
                    self.data.append((seq, target))

    def __getitem__(self, index):
        seq, target = self.data[index]
        return torch.tensor(seq, dtype=torch.float), torch.tensor(target, dtype=torch.float).view(1)

    def __len__(self):
        return len(self.data)


def get_all_data(file_path):
    df = pd.read_csv(file_path)

    print(f"数据文件: {file_path}")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    if len(df.columns) < 6:
        raise ValueError(f"数据需要至少6列，但只有{len(df.columns)}列")

    feature_columns = df.columns[1:6]  # 第2-6列是特征（油量+4个特征）
    target_column = df.columns[0]  # 第1列是目标效率值（%）

    # 归一化
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(df[feature_columns])
    target_scaled = target_scaler.fit_transform(df[[target_column]])

    full_data = np.hstack((target_scaled, features_scaled))
    original_targets = df[target_column].values

    return full_data, feature_scaler, target_scaler, original_targets, df[feature_columns].values


def split_data_by_interval(full_data, interval=10):
    total_samples = len(full_data) - 6
    all_indices = list(range(total_samples))
    test_indices = all_indices[::interval]
    train_indices = [i for i in all_indices if i not in test_indices]

    print(f"总样本数: {total_samples}")
    print(f"训练集样本数: {len(train_indices)}")
    print(f"测试集样本数: {len(test_indices)}")
    print(f"测试集抽取间隔: {interval}")

    return train_indices, test_indices


def get_train_test_data(file_path, interval=10):
    full_data, feature_scaler, target_scaler, original_targets, original_features = get_all_data(file_path)
    train_indices, test_indices = split_data_by_interval(full_data, interval)

    train_dataset = CustomDataset(full_data, train_indices)
    test_dataset = CustomDataset(full_data, test_indices)

    # 保存归一化器和原始数据
    train_dataset.feature_scaler = feature_scaler
    train_dataset.target_scaler = target_scaler
    test_dataset.feature_scaler = feature_scaler
    test_dataset.target_scaler = target_scaler

    # 保存原始特征值（用于功率计算）
    test_original_indices = [i + 6 for i in test_indices]
    test_dataset.original_features = original_features[test_original_indices]
    test_dataset.original_targets = original_targets[test_original_indices]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader