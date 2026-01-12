import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.h5_helpers import extract_hidden_features


class JetDataset(Dataset):
    def __init__(self, filepath, indices=None, input_dim=None, key="Jets", pt_cut=None, scaling_file=None):
        """
        JetDataset for loading jet data from HDF5 files, with dynamic feature scaling.

        Args:
            filepath: Path to HDF5 file, or list of paths for multi-file loading
            indices: Optional array of indices to use (for train/val split)
            input_dim: Number of input features to use
            key: HDF5 dataset key (default: "Jets")
            pt_cut: Optional pT threshold for filtering
            scaling_file: Path to .npz file containing 'mean' and 'std' arrays for scaling (REQUIRED)
        """
        if scaling_file is None:
            raise ValueError("scaling_file argument is required (path to .npz file with 'mean' and 'std').")
        print(f"Loading scaling from {scaling_file}")
        scaling_stats = np.load(scaling_file)
        self.scaling_mean = scaling_stats["mean"]
        self.scaling_std = scaling_stats["std"]

        # Handle single file or list of files
        if isinstance(filepath, (list, tuple)):
            self._load_multiple_files(filepath, key, pt_cut, indices)
        else:
            self._load_single_file(filepath, key, pt_cut, indices)
        self.input_dim = input_dim

    def _load_single_file(self, filepath, key, pt_cut, indices):
        """Load data from a single HDF5 file."""
        self.files = [h5py.File(filepath, 'r')]
        self.jets_list = [self.files[0][key]]

        self.pt = self.jets_list[0]['pt'][:]
        self.mass = self.jets_list[0]['mass'][:]
        self.gloParT_QCD = self.jets_list[0]['globalParT3_QCD'][:]
        self.gloParT_Tbqq = self.jets_list[0]['globalParT3_TopbWqq'][:]
        self.gloParT_Tbq = self.jets_list[0]['globalParT3_TopbWq'][:]

        self.total_len = len(self.jets_list[0])
        print(f"Loaded {filepath} with {self.total_len} jets")

        # Apply pt cut if requested
        if pt_cut is not None:
            selected = np.where(self.pt > pt_cut)[0]
            if indices is not None:
                indices = np.intersect1d(indices, selected, assume_unique=True)
            else:
                indices = selected

        self.indices = np.arange(self.total_len) if indices is None else np.array(indices)
        # Single file: direct indexing (file_idx always 0)
        self.file_indices = np.zeros(self.total_len, dtype=np.int32)
        self.local_indices = np.arange(self.total_len, dtype=np.int32)

    def _load_multiple_files(self, filepaths, key, pt_cut, indices):
        """Load data from multiple HDF5 files with balanced sampling.

        Always takes equal number of jets from each file (limited by smallest file).
        Uses consistent random sampling for reproducibility across train/val splits.
        """
        self.files = []
        self.jets_list = []

        all_pt = []
        all_mass = []
        all_gloParT_QCD = []
        all_gloParT_Tbqq = []
        all_gloParT_Tbq = []

        file_lengths = []

        # First pass: open files and get lengths
        for filepath in filepaths:
            f = h5py.File(filepath, 'r')
            self.files.append(f)
            jets = f[key]
            self.jets_list.append(jets)
            file_lengths.append(len(jets))
            print(f"Loaded {filepath} with {len(jets)} jets")

        # Determine sampling strategy
        # Always balance: take min(file_lengths) from each file with consistent seeding
        n_per_file = min(file_lengths)
        if indices is not None:
            print(f"Loading with train/val indices: using {n_per_file} jets from each of {len(filepaths)} files (consistent with parent)")
        else:
            print(f"Balancing: using {n_per_file} jets from each of {len(filepaths)} files")

        # Second pass: load data with consistent sampling
        for file_idx, (filepath, jets) in enumerate(zip(filepaths, self.jets_list)):
            # Use reproducible sampling based on file index
            np.random.seed(42 + file_idx)
            file_indices = np.random.choice(len(jets), size=n_per_file, replace=False)

            # Load metadata for selected jets
            all_pt.append(jets['pt'][:][file_indices])
            all_mass.append(jets['mass'][:][file_indices])
            all_gloParT_QCD.append(jets['globalParT3_QCD'][:][file_indices])
            all_gloParT_Tbqq.append(jets['globalParT3_TopbWqq'][:][file_indices])
            all_gloParT_Tbq.append(jets['globalParT3_TopbWq'][:][file_indices])

        # Concatenate all arrays
        self.pt = np.concatenate(all_pt)
        self.mass = np.concatenate(all_mass)
        self.gloParT_QCD = np.concatenate(all_gloParT_QCD)
        self.gloParT_Tbqq = np.concatenate(all_gloParT_Tbqq)
        self.gloParT_Tbq = np.concatenate(all_gloParT_Tbq)

        # Create mapping: global_index -> (file_index, local_index)
        self.file_indices = []
        self.local_indices = []

        actual_lengths = [len(all_pt[i]) for i in range(len(filepaths))]
        for file_idx, length in enumerate(actual_lengths):
            self.file_indices.extend([file_idx] * length)
            self.local_indices.extend(range(length))

        self.file_indices = np.array(self.file_indices, dtype=np.int32)
        self.local_indices = np.array(self.local_indices, dtype=np.int32)

        self.total_len = len(self.pt)

        # Apply pt cut if requested
        if pt_cut is not None:
            selected = np.where(self.pt > pt_cut)[0]
            if indices is not None:
                indices = np.intersect1d(indices, selected, assume_unique=True)
            else:
                indices = selected

        self.indices = np.arange(self.total_len) if indices is None else np.array(indices)
        if indices is not None:
            print(f"Applied {len(self.indices)}/{self.total_len} indices for train/val split")
        else:
            print(f"Total: {len(self.indices)} jets from {len(filepaths)} files")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get jet at index idx, with dynamic scaling applied."""
        global_idx = self.indices[idx]
        file_idx = self.file_indices[global_idx]
        local_idx = self.local_indices[global_idx]

        # Get jet from appropriate file
        jet = self.jets_list[file_idx][local_idx]
        features = extract_hidden_features(jet[None])[0].astype(np.float32)

        if self.input_dim is not None:
            features = features[:self.input_dim]
            mean = self.scaling_mean[:self.input_dim]
            std = self.scaling_std[:self.input_dim]
        else:
            mean = self.scaling_mean
            std = self.scaling_std

        features = (features - mean) / std
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
        """Close all open HDF5 files."""
        for f in self.files:
            f.close()
