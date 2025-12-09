import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.h5_helpers import extract_hidden_features
class JetDataset(Dataset):
    def __init__(self, filepath, indices=None, input_dim=None, key="Jets", pt_cut=None, pca_components=None):
        
        self.pca_components = None
        if pca_components is not None:
            if isinstance(pca_components, str):
                self.pca_components = np.load(pca_components).astype(np.float32)
            else:
                # allow passing np array directly
                self.pca_components = np.array(pca_components, dtype=np.float32)
            self.pca_components = self.pca_components[:input_dim]#Use only input_dim PCA components as input features

        self.file = h5py.File(filepath, 'r')
        self.jets = self.file[key]

        self.pt = self.jets['pt'][:]
        self.mass = self.jets['mass'][:]
        self.gloParT_QCD = self.jets['globalParT3_QCD'][:]
        self.gloParT_Tbqq = self.jets['globalParT3_TopbWqq'][:]
        self.gloParT_Tbq = self.jets['globalParT3_TopbWq'][:]
        self.mass = self.jets['mass'][:]
        self.total_len = len(self.jets)
        print(f"Loaded {filepath} with {self.total_len} jets")
        # Apply pt cut if requested
        if pt_cut is not None:
            selected = np.where(self.pt > pt_cut)[0]
            if indices is not None:
                indices = np.intersect1d(indices, selected, assume_unique=True)
            else:
                indices = selected

        self.indices = np.arange(self.total_len) if indices is None else np.array(indices)
        self.input_dim = input_dim

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        jet = self.jets[self.indices[idx]]
        features = extract_hidden_features(jet[None])[0].astype(np.float32)

        if self.pca_components is not None:
            features = self.pca_components @ features
        elif self.input_dim is not None:
            features = features[:self.input_dim]

        features_tensor = torch.tensor(features)
        return features_tensor, features_tensor

    def get_pt(self):
        return self.pt[self.indices]

    def get_mass(self):
        return self.mass[self.indices]

    def get_gloParT_QCD(self):
        return self.gloParT_QCD[self.indices]

    def get_gloParT_Tbqq(self):
        return self.gloParT_Tbqq[self.indices]

    def get_gloParT_Tbq(self):
        return self.gloParT_Tbq[self.indices]

    def close(self):
        self.file.close()
