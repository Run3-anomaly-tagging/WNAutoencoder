import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.h5_helpers import extract_hidden_features

class JetDataset(Dataset):
    def __init__(self, filepath, indices=None, input_dim=None, key="Jets"):
        self.file = h5py.File(filepath, 'r')
        self.jets = self.file[key]

        self.total_len = len(self.jets)
        self.indices = np.arange(self.total_len) if indices is None else np.array(indices)

        self.input_dim = input_dim  # max features to return, None means all

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        jet = self.jets[self.indices[idx]]
        features = extract_hidden_features(jet[None])[0].astype(np.float32)

        if self.input_dim is not None:
            features = features[:self.input_dim]

        features_tensor = torch.tensor(features)

        return features_tensor, features_tensor  # x == y for autoencoder

    def close(self):
        self.file.close()

