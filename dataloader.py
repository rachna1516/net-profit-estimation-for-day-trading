import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_utils import create_sequences, load_data


def lstm_create_sequences(data, seq_length):
    """Backwards-compatible alias: targets the first column (Open price)."""
    return create_sequences(data, seq_length, target_col=0)


class StockDataset(Dataset):
    """Wraps (features, targets) arrays as float32 tensors for a DataLoader."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        if len(features) != len(targets):
            raise ValueError(
                f"features and targets must have the same length "
                f"({len(features)} != {len(targets)})"
            )
        self.features = torch.tensor(np.asarray(features), dtype=torch.float32)
        self.targets = torch.tensor(np.asarray(targets), dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


if __name__ == "__main__":
    df = load_data("data.csv")
    print(df)
    X, y = lstm_create_sequences(df.values, 10)
    loader = DataLoader(StockDataset(X, y), batch_size=32, shuffle=False)
    features, targets = next(iter(loader))
    print(f"{len(loader.dataset)} samples; first batch X={tuple(features.shape)}, y={tuple(targets.shape)}")
