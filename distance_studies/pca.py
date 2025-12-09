import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from sklearn.decomposition import IncrementalPCA
from utils.jet_dataset import JetDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_explained_variance(output_dir, top_n=50, save_plots=True):
    components = np.load(os.path.join(output_dir, "components.npy"))
    explained_variance = np.load(os.path.join(output_dir, "explained_variance.npy"))
    explained_variance_ratio = np.load(os.path.join(output_dir, "explained_variance_ratio.npy"))
    
    n_components = len(explained_variance_ratio)
    top_n = min(top_n, n_components)
    
    plt.figure(figsize=(8,5))
    plt.bar(range(1, top_n+1), explained_variance_ratio[:top_n], alpha=0.7)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title(f"Explained Variance Ratio of Top {top_n} PCs")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "explained_variance_ratio.png"))
    
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, top_n+1), cumulative_variance[:top_n], marker='o')
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"Cumulative Explained Variance of Top {top_n} PCs")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=0.9, color='r', linestyle='--', label="90% variance")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "cumulative_explained_variance.png"))

def plot_pca_distributions(batch, components, output_dir, plot_name, n_plot=8):
    batch_np = batch[0].numpy() if isinstance(batch, (list, tuple)) else batch.numpy()
    X_pca = (components @ batch_np.T).T

    n_plot = min(n_plot, components.shape[0])
    n_rows = (n_plot + 1) // 2  # 2 per row
    plt.figure(figsize=(12, 3 * n_rows))
    bins = 50

    for i in range(n_plot):
        plt.subplot(n_rows, 2, i+1)
        plt.hist(X_pca[:, i], bins=bins, alpha=0.7, color='C'+str(i))
        plt.xlabel(f"PC{i+1} value")
        plt.ylabel("Counts")
        plt.title(f"PC{i+1} distribution")
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, plot_name))


def sample_batch(dataset: JetDataset, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset, replacement=True), pin_memory=True)
    batch = next(iter(loader))[0]
    return batch

def plot_correlation_matrix(batch, output_dir, plot_name="correlation_matrix.png"):
    X = batch[0].numpy() if isinstance(batch, (list, tuple)) else batch.numpy()

    def summarize_correlation(corr):
        # Exclude diagonal
        off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
        mean_abs = np.mean(np.abs(off_diag))
        max_abs = np.max(np.abs(off_diag))
        print(f"Mean |corr|: {mean_abs:.4f}")
        print(f"Max  |corr|: {max_abs:.4f}")
        return off_diag

    corr = np.corrcoef(X, rowvar=False)  # shape (D, D)
    off_diag = summarize_correlation(corr)

    # Plot 1: Correlation matrix
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, cmap="viridis", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Feature Correlation Matrix")
    plt.xlabel("Feature index")
    plt.ylabel("Feature index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()
    
    # Plot 2: Distribution of correlation values (excluding diagonal)
    plt.figure(figsize=(10, 6))
    plt.hist(off_diag, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Correlation coefficient")
    plt.ylabel("Count")
    plt.title(f"Distribution of Correlations (n={len(off_diag)})\nMean |corr|: {np.mean(np.abs(off_diag)):.4f}, Max |corr|: {np.max(np.abs(off_diag)):.4f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_distribution.png"))
    plt.close()




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PCA on input dataset")
    parser.add_argument("--config", type=str, default="../data/dataset_config_small.json")
    parser.add_argument("--process", type=str, default="QCD")
    parser.add_argument("--batch_size", type=int, default=100000)
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--calculate_pca", action='store_true', help="Calculate PCA (default: False, just plot)")
    parser.add_argument("--output_dir", type=str, default="./pca_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.calculate_pca:
        with open(args.config, "r") as f:
            config = json.load(f)

        if args.process not in config:
            raise KeyError(f"Process {args.process} not found in config.")

        dataset_path = config[args.process]["path"]
        dataset = JetDataset(dataset_path, input_dim=256)
        n_batches = int(np.ceil(len(dataset) / args.batch_size))
        
        ipca = IncrementalPCA(n_components=args.n_components, batch_size=args.batch_size)

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

        print("Fitting Incremental PCA")
        for i, batch in enumerate(tqdm(loader, total=n_batches, desc="PCA batches")):
            batch_np = batch[0].numpy() if isinstance(batch, (list, tuple)) else batch.numpy()
            ipca.partial_fit(batch_np)

        print("Incremental PCA fitting complete.")
        
        # Compute standardized PCA components (unit variance)
        components_std = ipca.components_ / np.sqrt(ipca.explained_variance_[:, None])
        np.save(os.path.join(args.output_dir, "components_std.npy"), components_std)
        np.save(os.path.join(args.output_dir, "components.npy"), ipca.components_)
        np.save(os.path.join(args.output_dir, "explained_variance.npy"), ipca.explained_variance_)
        np.save(os.path.join(args.output_dir, "explained_variance_ratio.npy"), ipca.explained_variance_ratio_)

        print(f"PCA results saved in {args.output_dir}")


        top_n = min(10, args.n_components)
        print(f"Top {top_n} principal components:")
        for i in range(top_n):
            print(f"PC{i+1}: {ipca.explained_variance_ratio_[i]:.4f}")

        cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
        print(f"Cumulative variance explained by top {top_n} PCs: {cumulative_variance[top_n-1]:.4f}")

        print("\nNorms of top principal component vectors:")
        for i in range(top_n):
            comp_norm = np.linalg.norm(ipca.components_[i])
            print(f"PC{i+1} norm: {comp_norm:.4f}")
        dataset.close()
    else:
        components = np.load(os.path.join(args.output_dir, "components.npy"))
        components_std = np.load(os.path.join(args.output_dir, "components_std.npy"))
        plot_explained_variance(args.output_dir, top_n=50)

        with open(args.config, "r") as f:
            config = json.load(f)
        dataset_path = config[args.process]["path"]
        dataset = JetDataset(dataset_path, input_dim=256)
        batch = sample_batch(dataset, batch_size=args.batch_size)
        #plot_pca_distributions(batch, components, args.output_dir, plot_name="pca_distributions.png", n_plot=8)
        #plot_pca_distributions(batch, components_std, args.output_dir, plot_name="pca_distributions_std.png", n_plot=8)
        batch = sample_batch(dataset, batch_size=args.batch_size)
        plot_correlation_matrix(batch, args.output_dir, "correlation_matrix.png")
        dataset.close()
