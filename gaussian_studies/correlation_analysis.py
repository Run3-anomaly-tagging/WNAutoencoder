"""
Correlation Analysis for Jet Features

This script computes and visualizes both linear (Pearson) and nonlinear (distance) 
correlations between jet features to understand feature dependencies.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import argparse
import dcor

import sys
sys.path.append('..')

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils.jet_dataset import JetDataset


def compute_pearson_correlation_matrix(X, subsample_size=None):
    """
    Compute Pearson correlation matrix for features in X.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        subsample_size: if provided, subsample for efficiency
    
    Returns:
        Correlation matrix of shape (n_features, n_features)
    """
    if subsample_size is not None and X.shape[0] > subsample_size:
        indices = np.random.choice(X.shape[0], subsample_size, replace=False)
        X = X[indices]
    
    print(f"Computing Pearson correlation on {X.shape[0]} samples, {X.shape[1]} features")
    corr_matrix = np.corrcoef(X, rowvar=False)
    
    return corr_matrix


def compute_distance_correlation_matrix(X, subsample_size=None):
    """
    Compute distance correlation matrix for features in X.
    
    Note: This can be computationally expensive for large datasets.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        subsample_size: if provided, subsample for efficiency
    
    Returns:
        Distance correlation matrix of shape (n_features, n_features)
    """
    if subsample_size is not None and X.shape[0] > subsample_size:
        indices = np.random.choice(X.shape[0], subsample_size, replace=False)
        X = X[indices]
    
    n_features = X.shape[1]
    dcor_matrix = np.zeros((n_features, n_features))
    
    print(f"Computing distance correlation on {X.shape[0]} samples, {n_features} features...")
    print("This may take a while...")
    
    # Compute pairwise distance correlations
    for i in tqdm(range(n_features), desc="Distance correlation progress"):
        for j in range(i, n_features):
            if i == j:
                dcor_matrix[i, j] = 1.0
            else:
                dcor_value = dcor.distance_correlation(X[:, i], X[:, j])
                dcor_matrix[i, j] = dcor_value
                dcor_matrix[j, i] = dcor_value
    
    return dcor_matrix


def plot_correlation_matrix(corr_matrix, output_dir, filename, title):
    """Plot correlation matrix as heatmap."""
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_matrix, cmap="viridis", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Feature index")
    plt.ylabel("Feature index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved {filename}")


def plot_correlation_distribution(corr_matrix, output_dir, filename, title):
    """Plot distribution of off-diagonal correlation values."""
    # Exclude diagonal
    n = corr_matrix.shape[0]
    off_diag = corr_matrix[~np.eye(n, dtype=bool)]
    
    mean_abs = np.mean(np.abs(off_diag))
    max_abs = np.max(np.abs(off_diag))
    
    plt.figure(figsize=(10, 6))
    plt.hist(off_diag, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Correlation coefficient")
    plt.ylabel("Count")
    plt.title(f"{title}\n(n={len(off_diag)} pairs) Mean |corr|: {mean_abs:.4f}, Max |corr|: {max_abs:.4f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved {filename}")
    
    return mean_abs, max_abs


def plot_comparison(pearson_matrix, distance_matrix, output_dir):
    """
    Plot comparison between Pearson and distance correlations.
    
    This helps identify nonlinear relationships: pairs with low Pearson 
    but high distance correlation indicate nonlinear dependencies.
    """
    n = pearson_matrix.shape[0]
    
    # Get off-diagonal values
    mask = ~np.eye(n, dtype=bool)
    pearson_vals = np.abs(pearson_matrix[mask])
    distance_vals = distance_matrix[mask]  # Already absolute since distance corr is always positive
    
    plt.figure(figsize=(10, 8))
    
    # Hexbin plot for density visualization
    plt.hexbin(pearson_vals, distance_vals, gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(label='Count')
    
    # Add diagonal line (where Pearson = distance correlation)
    max_val = max(pearson_vals.max(), distance_vals.max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Pearson = Distance')
    
    plt.xlabel("Absolute Pearson Correlation")
    plt.ylabel("Distance Correlation")
    #plt.title("Pearson vs Distance Correlation\n(Points above diagonal = nonlinear relationships)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pearson_vs_distance_correlation.png"))
    plt.close()
    print("Saved pearson_vs_distance_correlation.png")
    
    # Compute statistics on nonlinear relationships
    # Define nonlinear as: low Pearson but high distance correlation
    nonlinear_mask = (pearson_vals < 0.3) & (distance_vals > 0.5)
    n_nonlinear = nonlinear_mask.sum()
    pct_nonlinear = 100 * n_nonlinear / len(pearson_vals)
    
    print(f"\nNonlinear relationships detected:")
    print(f"  Pairs with |Pearson| < 0.3 but distance > 0.5: {n_nonlinear} ({pct_nonlinear:.2f}%)")


def sample_batch(dataset, batch_size):
    """Sample a batch from the dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, 
                       sampler=RandomSampler(dataset, replacement=True), 
                       pin_memory=True)
    batch = next(iter(loader))[0]
    return batch


def main():
    parser = argparse.ArgumentParser(
        description="Analyze linear and nonlinear correlations in jet features"
    )
    parser.add_argument("--config", type=str, default="data/dataset_config.json",
                       help="Path to dataset config JSON")
    parser.add_argument("--process", type=str, default="QCD",
                       help="Process name from config")
    parser.add_argument("--batch_size", type=int, default=10000,
                       help="Number of samples to use (default: 10k)")
    parser.add_argument("--output_dir", type=str, default="results/gaussian_studies/correlations",
                       help="Output directory for plots and results")
    parser.add_argument("--subsample_dcor", type=int, default=5000,
                       help="Subsample size for distance correlation (smaller=faster, default: 5k)")
    parser.add_argument("--load_distance_matrix", type=str, default=None,
                   help="Path to saved distance correlation matrix .npy file (skips computation)")
    args = parser.parse_args()

    
    # Create output directory
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset for process: {args.process}")
    config_path = os.path.join(project_root, args.config)
    with open(config_path, "r") as f:
        config = json.load(f)

    
    if args.process not in config:
        raise KeyError(f"Process {args.process} not found in config.")
    
    dataset_path = config[args.process]["path"]
    dataset = JetDataset(dataset_path, input_dim=256)
    
    # Sample data
    print(f"\nSampling {args.batch_size} jets...")
    batch = sample_batch(dataset, args.batch_size)
    X = batch.numpy() if hasattr(batch, 'numpy') else batch[0].numpy()
    print(f"Data shape: {X.shape}")
    
    # Compute Pearson correlation
    print("\n" + "="*60)
    print("PEARSON (LINEAR) CORRELATION ANALYSIS")
    print("="*60)
    pearson_matrix = compute_pearson_correlation_matrix(X)
    
    plot_correlation_matrix(
        pearson_matrix, args.output_dir, 
        "pearson_correlation_matrix.png",
        "Pearson Correlation Matrix (Linear)"
    )
    
    pearson_mean, pearson_max = plot_correlation_distribution(
        pearson_matrix, args.output_dir,
        "pearson_correlation_distribution.png",
        "Distribution of Pearson Correlations"
    )
    
    print(f"\nPearson correlation summary:")
    print(f"  Mean |correlation|: {pearson_mean:.4f}")
    print(f"  Max  |correlation|: {pearson_max:.4f}")
    
    # Compute Distance correlation
    print("\n" + "="*60)
    print("DISTANCE CORRELATION ANALYSIS (LINEAR + NONLINEAR)")
    print("="*60)
    if args.load_distance_matrix is not None:
        # Resolve path relative to project root
        dcor_matrix_path = os.path.join(project_root, args.load_distance_matrix)
        print(f"Loading distance correlation matrix from: {dcor_matrix_path}")
        distance_matrix = np.load(dcor_matrix_path)
        print(f"Loaded matrix shape: {distance_matrix.shape}")
    else:
        distance_matrix = compute_distance_correlation_matrix(X, subsample_size=args.subsample_dcor)
        np.save(os.path.join(output_dir, "distance_correlation_matrix.npy"), distance_matrix)
        print(f"Saved distance_correlation_matrix.npy")
    
    # Save distance correlation matrix (if not already saved)
    if args.load_distance_matrix is None:
        np.save(os.path.join(output_dir, "distance_correlation_matrix.npy"), distance_matrix)
        print(f"Saved distance_correlation_matrix.npy")
    
    plot_correlation_matrix(
        distance_matrix, output_dir,
        "distance_correlation_matrix.png",
        "Distance Correlation Matrix (Linear + Nonlinear)"
    )
    
    distance_mean, distance_max = plot_correlation_distribution(
        distance_matrix, output_dir,
        "distance_correlation_distribution.png",
        "Distribution of Distance Correlations"
    )
    
    print(f"\nDistance correlation summary:")
    print(f"  Mean correlation: {distance_mean:.4f}")
    print(f"  Max  correlation: {distance_max:.4f}")
    
    # Compare Pearson vs Distance
    print("\n" + "="*60)
    print("COMPARISON: PEARSON vs DISTANCE CORRELATION")
    print("="*60)
    
    # Need to recompute Pearson on same subsample for fair comparison
    if args.subsample_dcor < X.shape[0]:
        print(f"Recomputing Pearson on same {args.subsample_dcor} sample subset for comparison...")
        pearson_matrix_sub = compute_pearson_correlation_matrix(X, subsample_size=args.subsample_dcor)
    else:
        pearson_matrix_sub = pearson_matrix
    
    plot_comparison(pearson_matrix_sub, distance_matrix, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    dataset.close()


if __name__ == "__main__":
    main()
