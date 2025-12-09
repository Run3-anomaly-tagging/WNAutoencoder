import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler
from utils.jet_dataset import JetDataset
from model_config.model_registry import MODEL_REGISTRY
from wnae import WNAE


def sample_batch(dataset: JetDataset, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset, replacement=True), pin_memory=True)
    batch = next(iter(loader))[0]
    return batch

def load_distance_and_loss_data(distance_npz_path: str, models_base_dir: str, dims: list[int]):
    data = np.load(distance_npz_path)
    w_mean = data["w_mean"]
    w_std = data["w_std"]
    s_mean = data["s_mean"]
    s_std = data["s_std"]

    loss_by_dim = {}
    for dim in dims:
        ckpt_path = os.path.join(models_base_dir, f"paper_qcd_dim{dim}_wnae_PAPER", "checkpoint.pth")
        if not os.path.exists(ckpt_path):
            print(f"Warning: checkpoint for dim {dim} not found at {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        losses = ckpt.get("training_losses", [])
        if len(losses) > 0:
            loss_by_dim[dim] = losses[-1]
        else:
            print(f"Warning: no training_losses found in {ckpt_path}")

    return np.array(dims), w_mean, w_std, s_mean, s_std, loss_by_dim

def plot_distances(dims, wasserstein_means, wasserstein_stds, sinkhorn_means, sinkhorn_stds, output_dir: str):
    np.savez(os.path.join(output_dir, "distance_results.npz"),dims=dims, w_mean=wasserstein_means, w_std=wasserstein_stds,s_mean=sinkhorn_means, s_std=sinkhorn_stds)

    # Linear scale
    plt.figure()
    plt.errorbar(dims, wasserstein_means, yerr=wasserstein_stds, marker='o', label='Wasserstein')
    plt.errorbar(dims, sinkhorn_means, yerr=sinkhorn_stds, marker='s', label='Sinkhorn')
    plt.xlabel("Feature dimension")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "distances_linear.png"))
    plt.close()

    # Log scale
    plt.figure()
    plt.errorbar(dims, wasserstein_means, yerr=wasserstein_stds, marker='o', label='Wasserstein')
    plt.errorbar(dims, sinkhorn_means, yerr=sinkhorn_stds, marker='s', label='Sinkhorn')
    plt.xlabel("Feature dimension")
    plt.ylabel("Distance")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(output_dir, "distances_log.png"))
    plt.close()

def plot_combined_metrics(dims, w_mean, w_std, s_mean, s_std, loss_by_dim, output_dir: str):
    losses = [loss_by_dim.get(dim, np.nan) for dim in dims]

    plt.figure(figsize=(8, 5))
    plt.errorbar(dims, w_mean, yerr=w_std, marker='o', label='Wasserstein')
    plt.errorbar(dims, s_mean, yerr=s_std, marker='s', label='Sinkhorn')
    plt.plot(dims, losses, marker='^', linestyle='--', label='Final training loss', color='tab:red')
    plt.xlabel("Feature dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distances_vs_loss_linear.png"))
    plt.close()

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
    plt.savefig(os.path.join(output_dir, "distances_vs_loss_log.png"))
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Wasserstein and Sinkhorn distances on subsets from the same file")
    parser.add_argument("--config", type=str, default="../data/dataset_config_small.json")
    parser.add_argument("--process", type=str, default="QCD")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_pairs", type=int, default=10)
    parser.add_argument("--dims", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--output_dir", type=str, default="distance_plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r") as f:
        config = json.load(f)

    if args.process not in config:
        raise KeyError(f"Process {args.process} not found in config.")

    dataset_path = config[args.process]["path"]

    wasserstein_means = []
    wasserstein_stds = []
    sinkhorn_means = []
    sinkhorn_stds = []
    produce_npz = False
    if produce_npz:
        for dim in args.dims:
            print(f"Scanning dimension {dim}")
            dataset = JetDataset(dataset_path, input_dim=dim)
            wd_vals = []
            sd_vals = []

            for _ in range(args.n_pairs):
                batch1 = sample_batch(dataset, batch_size=args.batch_size)
                batch2 = sample_batch(dataset, batch_size=args.batch_size)
                wd_vals.append(WNAE.compute_wasserstein_distance(batch1, batch2).item())
                sd_vals.append(WNAE.compute_sinkhorn_divergence(batch1, batch2).item())

            print(f"Wasserstein: {wd_vals}")
            print(f"Sinkhorn: {sd_vals}")

            wasserstein_means.append(np.mean(wd_vals))
            wasserstein_stds.append(np.std(wd_vals))
            sinkhorn_means.append(np.mean(sd_vals))
            sinkhorn_stds.append(np.std(sd_vals))

            dataset.close()

        plot_distances(args.dims, wasserstein_means, wasserstein_stds, sinkhorn_means, sinkhorn_stds, args.output_dir)
    
    # Combine with training losses
    distance_npz_path = os.path.join(args.output_dir, "distance_results.npz")
    dims, w_mean, w_std, s_mean, s_std, loss_by_dim = load_distance_and_loss_data(distance_npz_path, "../models", args.dims)
    plot_combined_metrics(dims, w_mean, w_std, s_mean, s_std, loss_by_dim, args.output_dir)
