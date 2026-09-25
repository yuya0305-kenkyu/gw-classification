import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class AugmentedDataset(Dataset):
    """
    タイムシフト、振幅スケーリング、ガウスノイズ注入を施すデータセットクラス。
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray, transform: bool = False):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        x = self.data[idx].copy()
        y = self.labels[idx]

        if self.transform:
            # 1. タイムシフト (Time Shift: ±20ステップ)
            shift = np.random.randint(-20, 20)
            x = np.roll(x, shift, axis=-1)

            # 2. 振幅スケーリング (Amplitude Scaling: 0.9〜1.1倍)
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

            # 3. ノイズ注入 (Gaussian Noise Injection)
            noise = np.random.normal(0, 0.05, x.shape)
            x = x + noise

        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


def create_augmented_dataloader(df, arr: np.ndarray, labels: np.ndarray,
                                batch_size: int, training_size: int) -> dict:
    """
    訓練・検証セットをシャッフルなしの境界値で分割し、訓練側にデータ拡張を適用したDataLoaderを生成。
    """
    if len(arr) > 0:
        X = arr
    else:
        X = df.drop(['dec', 'ra', 'snr'], axis=1).values
    y = labels

    X_train = X[:training_size]
    X_val = X[training_size:]
    y_train = y[:training_size]
    y_val = y[training_size:]

    train_dataset = AugmentedDataset(X_train, y_train, transform=True)
    val_dataset = AugmentedDataset(X_val, y_val, transform=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {'train': train_loader, 'val': val_loader}


def create_test_dataloader(arr: np.ndarray, labels: np.ndarray,
                           n_sectors: int, batch_size: int) -> DataLoader:
    """
    テスト検証用のDataLoaderを生成。
    """
    X_test = torch.from_numpy(arr).float()
    y_test = torch.from_numpy(labels).long()
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return test_dataloader