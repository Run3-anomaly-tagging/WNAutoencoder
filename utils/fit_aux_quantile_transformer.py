"""
Utility to fit QuantileTransformer on auxiliary feature discriminants from training data.
Fits on mixed QCD+Top discriminants for all features together.

Usage:
    python utils/fit_aux_quantile_transformer.py
"""

import os
import sys
import numpy as np
import pickle
from sklearn.preprocessing import QuantileTransformer
import json
import h5py

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def fit_quantile_transformer(
    filepaths,
    aux_keys,
    output_path,
    n_quantiles=1000,
    output_distribution='normal',
    max_samples=100000
):
    """
    Fit QuantileTransformer on auxiliary feature discriminants from training files.
    
    Discriminants are normalized ratios: disc_i = x_i / sum(x_j) for each jet.
    Fits on mixed data from all training files (QCD + Top_bqq).
    
    Args:
        filepaths: List of HDF5 file paths (training data)
        aux_keys: List of auxiliary feature keys
        output_path: Where to save fitted transformer (.pkl)
        n_quantiles: Number of quantiles for transform (default: 1000)
        output_distribution: 'normal' (Gaussian) or 'uniform' (default: 'normal')
        max_samples: Maximum samples to use for fitting
    """
    print("=" * 80)
    print("FITTING QUANTILE TRANSFORMER FOR AUXILIARY DISCRIMINANTS")
    print("=" * 80)
    print(f"Files: {filepaths}")
    print(f"Auxiliary keys: {aux_keys}")
    print(f"Output: {output_path}")
    print(f"Settings: n_quantiles={n_quantiles}, output_distribution={output_distribution}")
    print("Discriminants: aux_i / sum(aux_j) [normalized ratios]")
    print("=" * 80)
    
    all_aux_data = {key: [] for key in aux_keys}
    
    samples_per_file = max_samples // len(filepaths)
    
    for filepath in filepaths:
        print(f"\nLoading {filepath}...")
        with h5py.File(filepath, 'r') as f:
            jets = f['Jets']
            n_jets = len(jets)

            if n_jets > samples_per_file:
                np.random.seed(42)
                indices = np.random.choice(n_jets, size=samples_per_file, replace=False)
            else:
                indices = np.arange(n_jets)
            
            for aux_key in aux_keys:
                if aux_key in jets.dtype.names:
                    aux_values = jets[aux_key][:][indices]
                    all_aux_data[aux_key].append(aux_values)
                    print(f"  {aux_key}: loaded {len(aux_values)} samples")
                else:
                    print(f"  ERROR: {aux_key} not found in {filepath}")
                    print(f"  Available fields: {jets.dtype.names}")
                    return None
    
    # Stack raw values into (n_samples, n_features) array
    aux_raw = np.column_stack([np.concatenate(all_aux_data[key]) for key in aux_keys]).astype(np.float32)
    print(f"\nCollected {aux_raw.shape[0]} samples x {aux_raw.shape[1]} features")
    print(f"Raw value ranges:")
    for i, key in enumerate(aux_keys):
        print(f"  {key}: [{aux_raw[:, i].min():.3f}, {aux_raw[:, i].max():.3f}] "
              f"(mean={aux_raw[:, i].mean():.3f}, std={aux_raw[:, i].std():.3f})")
    
    aux_sum = aux_raw.sum(axis=1, keepdims=True) + 1e-8
    discriminants = aux_raw / aux_sum
    
    print(f"\nDiscriminant ranges (should sum to 1.0):")
    for i, key in enumerate(aux_keys):
        print(f"  {key}: [{discriminants[:, i].min():.3f}, {discriminants[:, i].max():.3f}] "
              f"(mean={discriminants[:, i].mean():.3f}, std={discriminants[:, i].std():.3f})")
    print(f"  Sum check: mean={discriminants.sum(axis=1).mean():.6f} (should be ~1.0)")
    
    # Fit transformer on discriminants
    print("\nFitting QuantileTransformer on discriminants")
    print("  Mapping quantiles to Gaussian N(0,1)")
    
    transformer = QuantileTransformer(
        n_quantiles=min(n_quantiles, len(discriminants)),
        output_distribution=output_distribution,
        subsample=max_samples,
        random_state=42
    )
    transformer.fit(discriminants)
    
    # Validate transformation
    disc_transformed = transformer.transform(discriminants[:1000])
    print(f"\nTransformed value ranges (first 1000 samples, should be (mean, sigma)=(0,1)):")
    for i, key in enumerate(aux_keys):
        print(f"  {key}: [{disc_transformed[:, i].min():.3f}, {disc_transformed[:, i].max():.3f}] "
              f"(mean={disc_transformed[:, i].mean():.3f}, std={disc_transformed[:, i].std():.3f})")
    
    # Save transformer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(transformer, f)
    
    print(f"\n Saved transformer to: {output_path}")
    print("  Transformer maps: discriminants [0,1] -> Gaussian N(0,1)")
    print("=" * 80)
    
    return transformer


if __name__ == "__main__":
    CONFIG_PATH = os.path.join(project_root, "data", "dataset_config.json")
    
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    # Training files (QCD + Top_bqq backgrounds)
    TRAIN_FILES = [
        config["QCD"]["path"],
        config["Top_bqq"]["path"]
    ]
    
    AUX_KEYS = ['globalParT3_QCD', 'globalParT3_TopbWqq']
    OUTPUT_PATH = os.path.join(project_root, "data", "aux_quantile_transformer.pkl")
    
    fit_quantile_transformer(
        filepaths=TRAIN_FILES,
        aux_keys=AUX_KEYS,
        output_path=OUTPUT_PATH,
        n_quantiles=1000,
        output_distribution='normal',
        max_samples=100000
    )
