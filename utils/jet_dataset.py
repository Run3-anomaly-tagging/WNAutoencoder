import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.h5_helpers import extract_hidden_features
class JetDataset(Dataset):
    def __init__(self, filepath, indices=None, input_dim=None, key="Jets",
                 mean=None, std=None, pt_cut=None):
        self.file = h5py.File(filepath, 'r')
        self.jets = self.file[key]

        self.pt = self.jets['pt'][:]
        self.mass = self.jets['mass'][:]

        self.total_len = len(self.jets)

        # Apply pt cut if requested
        if pt_cut is not None:
            selected = np.where(self.pt > pt_cut)[0]
            if indices is not None:
                indices = np.intersect1d(indices, selected, assume_unique=True)
            else:
                indices = selected

        self.indices = np.arange(self.total_len) if indices is None else np.array(indices)

        self.input_dim = input_dim

        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        jet = self.jets[self.indices[idx]]
        features = extract_hidden_features(jet[None])[0].astype(np.float32)

        if self.input_dim is not None:
            features = features[:self.input_dim]

        features_tensor = torch.tensor(features)

        if self.mean is not None and self.std is not None:
            features_tensor = (features_tensor - self.mean[:len(features_tensor)]) / self.std[:len(features_tensor)]

        return features_tensor, features_tensor

    def get_pt(self):
        return self.pt[self.indices]

    def get_mass(self):
        return self.mass[self.indices]

    def close(self):
        self.file.close()
