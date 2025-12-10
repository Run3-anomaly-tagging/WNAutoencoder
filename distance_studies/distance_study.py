import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils.jet_dataset import JetDataset
from model_config.model_registry import MODEL_REGISTRY
from wnae import WNAE


def sample_batch(dataset: JetDataset, batch_size: int):
    """Sample a random batch from dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset, replacement=True), pin_memory=True)
    batch = next(iter(loader))[0]
    return batch

def load_distance_and_loss_data(distance_npz_path: str, models_base_dir: str, dims: list[int]):
    """Load precomputed distance statistics and final training losses from checkpoints."""
    data = np.load(distance_npz_path)
    w_mean = data["w_mean"]
    w_std = data["w_std"]
    s_mean = data["s_mean"]
    s_std = data["s_std"]

    loss_by_dim = {}
    for dim in dims:
        ckpt_path = os.path.join(models_base_dir, f"paper_qcd_dim{dim}_wnae_PAPER", "checkpoint.pth")
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] Checkpoint for dim {dim} not found at {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        losses = ckpt.get("training_losses", [])
        if len(losses) > 0:
            loss_by_dim[dim] = losses[-1]
        else:
            print(f"[WARNING] No training_losses found in {ckpt_path}")

    return np.array(dims), w_mean, w_std, s_mean, s_std, loss_by_dim

def plot_distances(dims, wasserstein_means, wasserstein_stds, sinkhorn_means, sinkhorn_stds, output_dir: str):
    """Save distance statistics and generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez(
        os.path.join(output_dir, "distance_results.npz"),
        dims=dims, 
        w_mean=wasserstein_means, 
        w_std=wasserstein_stds,
        s_mean=sinkhorn_means, 
        s_std=sinkhorn_stds
    )

    # Linear scale
    plt.figure()
    plt.errorbar(dims, wasserstein_means, yerr=wasserstein_stds, marker='o', label='Wasserstein')
    plt.errorbar(dims, sinkhorn_means, yerr=sinkhorn_stds, marker='s', label='Sinkhorn')
    plt.xlabel("Feature dimension")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distances_linear.png"), dpi=200)
    plt.close()
    print(f"[INFO] Saved: {os.path.join(output_dir, 'distances_linear.png')}")

    # Log scale
    plt.figure()
    plt.errorbar(dims, wasserstein_means, yerr=wasserstein_stds, marker='o', label='Wasserstein')
    plt.errorbar(dims, sinkhorn_means, yerr=sinkhorn_stds, marker='s', label='Sinkhorn')
    plt.xlabel("Feature dimension")
    plt.ylabel("Distance")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distances_log.png"), dpi=200)
    plt.close()
    print(f"[INFO] Saved: {os.path.join(output_dir, 'distances_log.png')}")

def plot_combined_metrics(dims, w_mean, w_std, s_mean, s_std, loss_by_dim, output_dir: str):
    """Plot distances alongside final training losses."""
    losses = [loss_by_dim.get(dim, np.nan) for dim in dims]

    # Linear scale
    plt.figure(figsize=(8, 5))
    plt.errorbar(dims, w_mean, yerr=w_std, marker='o', label='Wasserstein')
    plt.errorbar(dims, s_mean, yerr=s_std, marker='s', label='Sinkhorn')
    plt.plot(dims, losses, marker='^', linestyle='--', label='Final training loss', color='tab:red')
    plt.xlabel("Feature dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distances_vs_loss_linear.png"), dpi=200)
    plt.close()
    print(f"[INFO] Saved: {os.path.join(output_dir, 'distances_vs_loss_linear.png')}")

    # Log scale
    plt.figure(figsize=(8, 5))
    plt.errorbar(dims, w_mean, yerr=w_std, marker='o', label='Wasserstein')
    plt.errorbar(dims, s_mean, yerr=s_std, marker='s', label='Sinkhorn')
    plt.plot(dims, losses, marker='^', linestyle='--', label='Final training loss', color='tab:red')
    plt.xlabel("Feature dimension")
    plt.ylabel("Value")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distances_vs_loss_log.png"), dpi=200)
    plt.close()
    print(f"[INFO] Saved: {os.path.join(output_dir, 'distances_vs_loss_log.png')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Wasserstein and Sinkhorn distances across feature dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute distances from scratch
  python distance_studies/distance_study.py --compute --dims 8 16 32 64 128 256
  
  # Plot only (using existing distance_results.npz)
  python distance_studies/distance_study.py --dims 8 16 32 64 128 256
  
  # Custom dataset and output
  python distance_studies/distance_study.py --compute --config data/dataset_config.json --process QCD --output-dir results/distance_study_qcd
        """
    )
    parser.add_argument("--config", type=str, default="data/dataset_config.json", help="Path to dataset config JSON")
    parser.add_argument("--process", type=str, default="QCD", help="Process name from config")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for sampling")
    parser.add_argument("--n-pairs", type=int, default=10, help="Number of batch pairs to compare per dimension")
    parser.add_argument("--dims", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256], help="Feature dimensions to test")
    parser.add_argument("--output-dir", type=str, default="results/distance_study/main", help="Output directory for plots")
    parser.add_argument("--compute", action="store_true", help="Compute distances (otherwise just plot existing results)")
    args = parser.parse_args()

    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(project_root, args.config)
    with open(config_path, "r") as f:
        config = json.load(f)

    if args.process not in config:
        raise KeyError(f"Process '{args.process}' not found in config: {config_path}")

    dataset_path = config[args.process]["path"]

    if args.compute:
        print("[INFO] Computing distances from dataset...")
        wasserstein_means = []
        wasserstein_stds = []
        sinkhorn_means = []
        sinkhorn_stds = []

        for dim in args.dims:
            print(f"[INFO] Processing dimension {dim}...")
            dataset = JetDataset(dataset_path, input_dim=dim)
            wd_vals = []
            sd_vals = []

            for i in range(args.n_pairs):
                batch1 = sample_batch(dataset, batch_size=args.batch_size)
                batch2 = sample_batch(dataset, batch_size=args.batch_size)
                wd_vals.append(WNAE.compute_wasserstein_distance(batch1, batch2).item())
                sd_vals.append(WNAE.compute_sinkhorn_divergence(batch1, batch2).item())
                print(f"  Pair {i+1}/{args.n_pairs}: W={wd_vals[-1]:.4f}, S={sd_vals[-1]:.4f}")

            wasserstein_means.append(np.mean(wd_vals))
            wasserstein_stds.append(np.std(wd_vals))
            sinkhorn_means.append(np.mean(sd_vals))
            sinkhorn_stds.append(np.std(sd_vals))

            dataset.close()

        plot_distances(args.dims, wasserstein_means, wasserstein_stds, sinkhorn_means, sinkhorn_stds, output_dir)
    
    # Load distances and combine with training losses
    distance_npz_path = os.path.join(output_dir, "distance_results.npz")
    if not os.path.exists(distance_npz_path):
        print(f"[ERROR] Distance results not found: {distance_npz_path}")
        print("[INFO] Run with --compute flag to generate distances first.")
        sys.exit(1)
    
    models_base_dir = os.path.join(project_root, "models")
    dims, w_mean, w_std, s_mean, s_std, loss_by_dim = load_distance_and_loss_data(
        distance_npz_path, models_base_dir, args.dims
    )
    plot_combined_metrics(dims, w_mean, w_std, s_mean, s_std, loss_by_dim, output_dir)
    
    print(f"\n[INFO] All plots saved to: {output_dir}")
