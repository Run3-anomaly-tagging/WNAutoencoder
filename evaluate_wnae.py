import os
import json
import argparse
import warnings
from typing import Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, SequentialSampler
from matplotlib.transforms import Bbox

from utils.jet_dataset import JetDataset
from wnae import WNAE
from model_config.model_registry import MODEL_REGISTRY
from model_config.model_config import WNAE_PARAM_PRESETS

def compute_mse(model: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    """Compute per-sample MSE (energy) for dataset using model encoder/decoder."""
    model.eval()
    mses = []
    for batch in dataloader:
        x = batch[0].to(device)
        recon_x = model.decoder(model.encoder(x))
        per_sample_mse = torch.mean((x - recon_x) ** 2, dim=1)
        mses.extend(per_sample_mse.detach().cpu().numpy())
    return np.array(mses)

def plot_checkpoint_energies(checkpoint: Dict[str, Any], plot_dir="plots"):
    """
    Plot positive and negative energies per batch from a loaded checkpoint.
    """
    os.makedirs(plot_dir, exist_ok=True)

    pos_energies = checkpoint.get("batch_pos_energies", None)
    neg_energies = checkpoint.get("batch_neg_energies", None)

    if pos_energies is None or neg_energies is None:
        print("Checkpoint keys:", checkpoint.keys())
        warnings.warn("[WARNING] Positive/Negative energies not found in checkpoint. Skipping energy-per-batch plot.")
        return

    pos_energies = np.array(pos_energies)
    neg_energies = np.array(neg_energies)

    plt.figure(figsize=(8, 5))
    plt.plot(pos_energies, label="Positive Energy")
    plt.plot(neg_energies, label="Negative Energy")
    plt.xlabel("Batch number")
    plt.ylabel("Energy")
    plt.legend(frameon=False)
    plt.yscale('log')
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "energies_per_batch.png")
    plt.savefig(plot_path)
    plt.savefig(plot_path.replace(".png", ".pdf"))
    plt.close()
    print(f"[INFO] Energy plot saved to: {plot_path}")

def load_dataset(file_path: str, input_dim: int, key="Jets", max_jets=10000, pt_cut=None):
    """
    Load JetDataset and subsample to max_jets if needed.
    """
    tmp_ds = JetDataset(file_path, input_dim=input_dim, key=key, pt_cut=pt_cut)
    if len(tmp_ds) > max_jets:
        sampled = np.random.choice(tmp_ds.indices, size=max_jets, replace=False)
        tmp_ds.indices = sampled
    return tmp_ds

def plot_eff_vs_pt(bkg_mses: np.ndarray, sig_mses_dict: Dict[str, np.ndarray],
                   bkg_dataset: JetDataset, signal_loaders: Dict[str, DataLoader],
                   wp=0.1, savedir="plots", bkg_name="background"):
    """
    Plot efficiency vs jet pT for a fixed working point defined by a background mistag rate.
    """
    os.makedirs(savedir, exist_ok=True)
    # threshold from background: WP corresponds to (1 - wp) quantile
    threshold = np.percentile(bkg_mses, 100 * (1 - wp))

    bins_pt = np.linspace(150, 800, 50)
    bin_centers = 0.5 * (bins_pt[:-1] + bins_pt[1:])

    # background efficiency per pT bin
    bkg_pts = bkg_dataset.get_pt()
    bkg_eff_pt = []
    for i in range(len(bins_pt) - 1):
        mask = (bkg_pts >= bins_pt[i]) & (bkg_pts < bins_pt[i + 1])
        if np.sum(mask) > 0:
            eff = np.mean(bkg_mses[mask] > threshold)
        else:
            eff = np.nan
        bkg_eff_pt.append(eff)

    # signal efficiencies per pT bin
    sig_eff_pt_dict = {}
    for name, sig_mses in sig_mses_dict.items():
        sig_pts = signal_loaders[name].dataset.get_pt()
        sig_eff = []
        for i in range(len(bins_pt) - 1):
            mask = (sig_pts >= bins_pt[i]) & (sig_pts < bins_pt[i + 1])
            if np.sum(mask) > 0:
                eff = np.mean(sig_mses[mask] > threshold)
            else:
                eff = np.nan
            sig_eff.append(eff)
        sig_eff_pt_dict[name] = sig_eff

    # plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_centers, bkg_eff_pt, label=f"{bkg_name} mistag (WP={wp*100:.0f}%)", linestyle="--")
    for name, eff in sig_eff_pt_dict.items():
        ax.plot(bin_centers, eff, label=f"{name}")
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0, 1.3)
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig = os.path.join(savedir, f"eff_vs_pt_wp_{wp}.png")
    plt.savefig(savefig, dpi=200)
    plt.close()
    print(f"[INFO] Saved {savefig}")

def plot_sample_vs_reconstruction(model: WNAE, bkg_loader: DataLoader, savedir: str,
                                  device: torch.device = torch.device("cpu")):
    """
    Plot one MCMC jet and one validation jet with original and reconstructed features.
    """
    os.makedirs(savedir, exist_ok=True)
    try:
        val_batch = next(iter(bkg_loader))
    except StopIteration:
        warnings.warn("Background loader is empty; skipping sample_vs_reconstruction plot.")
        return
    val_jet = val_batch[0][0:1].to(device)  # first jet in batch

    try:
        val_energy, val_z, val_reco = model._WNAE__energy_with_samples(val_jet)
    except Exception as e:
        warnings.warn(f"Failed to compute energy_with_samples for val jet: {e}")
        return

    # Get one MCMC jet
    if not hasattr(model, "buffer") or len(model.buffer.buffer) == 0:
        warnings.warn("MCMC buffer is empty; skipping MCMC sample in sample_vs_reconstruction.")
        # still plot val jet vs its reconstruction only
        mcmc_jet = None
        mcmc_reco = None
        mcmc_energy = None
        mcmc_z = None
    else:
        mcmc_jet = model.buffer.buffer[0].unsqueeze(0).to(device)
        try:
            mcmc_energy, mcmc_z, mcmc_reco = model._WNAE__energy_with_samples(mcmc_jet)
        except Exception as e:
            warnings.warn(f"Failed to compute energy_with_samples for MCMC jet: {e}")
            mcmc_jet = None
            mcmc_reco = None
            mcmc_energy = None
            mcmc_z = None

    n_features = val_jet.shape[1]
    features = range(n_features)

    plt.figure(figsize=(10, 5))
    plt.plot(features, val_jet[0].cpu().numpy(), 'o-', label='Val jet')
    plt.plot(features, val_reco[0].detach().cpu().numpy(), 's--', label='Val jet reco.')

    if mcmc_jet is not None:
        plt.plot(features, mcmc_jet[0].cpu().numpy(), 'o-', label='MCMC jet')
        plt.plot(features, mcmc_reco[0].detach().cpu().numpy(), 's--', label='MCMC jet reco.')

    plt.xlabel("Feature index")
    plt.ylabel("Feature value")

    if val_energy is not None:
        plt.text(0.95, 0.95, f"Val energy: {val_energy.item():.1f}", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='top')

    if 'mcmc_energy' in locals() and mcmc_energy is not None:
        plt.text(0.95, 0.90, f"MCMC energy: {mcmc_energy.item():.1f}", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='top')

    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(savedir, "sample_reconstruction.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved {save_path}")

    # Latent plot (if available)
    if 'val_z' in locals() and val_z is not None:
        plt.figure(figsize=(10, 5))
        n_latent = val_z.shape[1]
        latent_idx = range(n_latent)
        plt.plot(latent_idx, val_z[0].detach().cpu().numpy(), 'o-', label='Val jet z')
        if 'mcmc_z' in locals() and mcmc_z is not None:
            plt.plot(latent_idx, mcmc_z[0].detach().cpu().numpy(), 'o-', label='MCMC jet z')
        plt.xlabel("Latent dimension index")
        plt.ylabel("Latent value")
        plt.legend()
        plt.tight_layout()
        save_path_z = os.path.join(savedir, "sample_latent.png")
        plt.savefig(save_path_z)
        plt.close()
        print(f"[INFO] Saved {save_path_z}")

def plot_energy_distributions(model: WNAE, bkg_loader: DataLoader, n_samples=10000,
                              savedir="plots", device: torch.device = torch.device("cpu")):
    """
    Plot distributions of positive (data) and negative (MCMC) reconstruction energies.
    """
    os.makedirs(savedir, exist_ok=True)
    model.eval()

    E_pos_list = []
    E_neg_list = []

    # Collect positive (data) energies
    count = 0
    with torch.no_grad():
        for batch in bkg_loader:
            jets = batch[0].to(device)
            for jet in jets:
                try:
                    energy, _, _ = model._WNAE__energy_with_samples(jet.unsqueeze(0))
                except Exception as e:
                    warnings.warn(f"Failed to compute energy for a data jet: {e}")
                    continue
                E_pos_list.append(energy.item())
                count += 1
                if count >= n_samples:
                    break
            if count >= n_samples:
                break

    # Collect negative (MCMC) energies
    if not hasattr(model, "buffer") or len(model.buffer.buffer) == 0:
        warnings.warn("MCMC buffer empty; skipping E- distribution.")
    else:
        for i in range(min(n_samples, len(model.buffer.buffer))):
            mcmc_jet = model.buffer.buffer[i].unsqueeze(0).to(device)
            try:
                energy, _, _ = model._WNAE__energy_with_samples(mcmc_jet)
                E_neg_list.append(energy.item())
            except Exception as e:
                warnings.warn(f"Failed to compute energy for MCMC jet: {e}")

    # If no data or mcmc energies were collected, warn and return
    if len(E_pos_list) == 0:
        warnings.warn("No positive energies collected; skipping energy distributions plot.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    x_pos_max = np.percentile(E_pos_list, 99)
    x_neg_max = np.percentile(E_neg_list, 99)

    bins_pos = np.linspace(0, x_pos_max, 50)
    bins_neg = np.linspace(0, x_neg_max, 50)

    axs[0].hist(E_pos_list, bins=bins_pos,histtype='step', color='C0', fill=False)
    axs[0].set_title("E+ (data)")
    axs[0].set_xlabel("Reconstruction energy")
    axs[0].set_ylabel("Counts")

    if len(E_neg_list) > 0:
        axs[1].hist(E_neg_list, bins=bins_neg,histtype='step', color='C1', fill=False)
    axs[1].set_title("E- (MCMC)")
    axs[1].set_xlabel("Reconstruction energy")

    axs[0].set_xlim(0, x_pos_max)
    axs[1].set_xlim(0, x_neg_max)

    plt.tight_layout()
    save_path = os.path.join(savedir, "energy_distributions.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved {save_path}")

def plot_reco_map_with_mcmc(model: WNAE, savedir="plots", n_bins=20,
                            n_chains_to_plot=1, feature_idx=(0, 1),
                            value_range=(-4, 4), device: torch.device = torch.device("cpu"), d_min=1.0):
    """
    Plot a 2D map of model energies over buffer samples, overlaid with MCMC chains.
    """
    os.makedirs(savedir, exist_ok=True)
    f1, f2 = feature_idx
    vmin, vmax = value_range

    if not hasattr(model, "buffer") or len(model.buffer.buffer) == 0:
        warnings.warn("Model buffer is empty or missing. Skipping reco map with mcmc.")
        return

    mcmc_jet = torch.stack(model.buffer.buffer).to(device)
    try:
        energy = model._WNAE__energy(mcmc_jet).detach().cpu().numpy()
    except Exception as e:
        warnings.warn(f"Failed to compute energies for buffer stack: {e}")
        return

    x = mcmc_jet[:, f1].cpu().numpy()
    y = mcmc_jet[:, f2].cpu().numpy()

    bin_edges = np.linspace(vmin, vmax, n_bins + 1)
    hist_sum, _, _ = np.histogram2d(x, y, bins=(bin_edges, bin_edges), weights=energy)
    hist_count, _, _ = np.histogram2d(x, y, bins=(bin_edges, bin_edges))
    avg_energy = np.divide(hist_sum, hist_count, out=np.zeros_like(hist_sum), where=hist_count > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.where(hist_count.T > 0, avg_energy.T, np.nan), origin="lower",
                   extent=[vmin, vmax, vmin, vmax],
                   aspect="auto", cmap="plasma")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Reco. error")
    ax.set_xlabel(f"Feature {f1}")
    ax.set_ylabel(f"Feature {f2}")

    # --- Overlay MCMC trajectories ---
    def select_initial_indices(x_arr, y_arr, n, d_min_local):
        """
        Greedy selection of indices such that each selected point is at least d_min away from others.
        """
        selected = []
        for idx in np.random.permutation(len(x_arr)):
            if len(selected) == 0:
                selected.append(idx)
                if len(selected) >= n:
                    break
                continue
            dist = np.sqrt((x_arr[idx] - x_arr[selected]) ** 2 + (y_arr[idx] - y_arr[selected]) ** 2)
            if np.all(dist >= d_min_local):
                selected.append(idx)
                if len(selected) >= n:
                    break
        return selected

    start_indices = select_initial_indices(x, y, n_chains_to_plot, d_min)
    if len(start_indices) == 0:
        warnings.warn("Could not select any initial indices for overlaying MCMC chains.")
    else:
        init_x = mcmc_jet[start_indices]
        try:
            chains = model.run_mcmc(x=init_x, all_steps=True).detach().cpu().numpy()
        except Exception as e:
            warnings.warn(f"Failed to run_mcmc for selected init_x: {e}")
            chains = None

        if chains is not None:
            for i in range(min(n_chains_to_plot, chains.shape[0])):
                traj = chains[i]
                f1_vals = traj[f1, :]
                f2_vals = traj[f2, :]
                if i == 0:
                    ax.plot(f1_vals, f2_vals, "ko-", markersize=4, linewidth=1., label="MCMC steps")
                    ax.plot(f1_vals[0], f2_vals[0], "ro", markersize=4, label="Initial samples")
                    ax.plot(f1_vals[-1], f2_vals[-1], "bo", markersize=4, label="Final samples")
                else:
                    ax.plot(f1_vals, f2_vals, "ko-", markersize=4, linewidth=1.)
                    ax.plot(f1_vals[0], f2_vals[0], "ro", markersize=4)
                    ax.plot(f1_vals[-1], f2_vals[-1], "bo", markersize=4)

    ax.legend()
    plt.tight_layout()
    outpath = os.path.join(savedir, "reco_energy_map_mcmc.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[INFO] Saved {outpath}")

def make_summary_plots(
    bkg_mses,
    sig_mses_dict,
    bkg_dataset,
    signal_loaders,
    summary,
    savedir,
    bkg_name="QCD",
):
    """Create and save summary plots: MSE, ROC, jet mass, and jet pt."""

    os.makedirs(os.path.join(savedir, "plots"), exist_ok=True)

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ---- 1. MSE distributions ----
        ax_mse = axes[0, 0]
        all_mses = np.concatenate([bkg_mses] + list(sig_mses_dict.values())) if len(sig_mses_dict) > 0 else bkg_mses
        _, x_max = np.percentile(all_mses, [0, 99.]) if len(all_mses) > 1 else (0, max(all_mses) if len(all_mses) else 1.0)
        bins_mse = np.linspace(0, x_max, 101)
        ax_mse.hist(bkg_mses, bins=bins_mse, histtype='step', label=bkg_name, density=True)
        for name, mses in sig_mses_dict.items():
            ax_mse.hist(mses, bins=bins_mse, histtype='step', label=name, density=True)
        ax_mse.set_xlabel("Reconstruction MSE")
        ax_mse.set_ylabel("Density")
        ax_mse.set_xlim([0, x_max])
        ax_mse.legend()

        # ---- 2. ROC curves ----
        ax_roc = axes[0, 1]
        for name, sig_mses in sig_mses_dict.items():
            labels = np.concatenate([np.zeros_like(bkg_mses), np.ones_like(sig_mses)])
            scores = np.concatenate([bkg_mses, sig_mses])
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            summary.setdefault("aucs", {})[name] = float(roc_auc)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        ax_roc.set_xlabel("Background mistag rate")
        ax_roc.set_ylabel("Signal efficiency")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, alpha=0.3)

        # ---- 3. Jet mass distributions ----
        ax_mass = axes[1, 0]
        bins_mass = np.linspace(0, 200, 101)
        ax_mass.hist(bkg_dataset.get_mass(), bins=bins_mass, histtype='step', density=True, label=bkg_name)
        for name, loader in signal_loaders.items():
            sig_ds = loader.dataset
            ax_mass.hist(sig_ds.get_mass(), bins=bins_mass, histtype='step', density=True, label=name)
        ax_mass.set_xlabel("Jet mass [GeV]")
        ax_mass.set_ylabel("Density")
        ax_mass.legend()

        # ---- 4. Jet pt distributions ----
        ax_pt = axes[1, 1]
        bins_pt = np.linspace(150, 800, 65)
        ax_pt.hist(bkg_dataset.get_pt(), bins=bins_pt, histtype='step', density=True, label=bkg_name)
        for name, loader in signal_loaders.items():
            sig_ds = loader.dataset
            ax_pt.hist(sig_ds.get_pt(), bins=bins_pt, histtype='step', density=True, label=name)
        ax_pt.set_xlabel("Jet $p_T$ [GeV]")
        ax_pt.set_ylabel("Density")
        ax_pt.set_ylim(1e-4, 3e-2)
        ax_pt.set_yscale("log")
        ax_pt.legend()

        # ---- Save combined figure ----
        plt.tight_layout()
        savefig = os.path.join(savedir, "plots", "summary.png")
        plt.savefig(savefig, dpi=200)
        plt.close(fig)
        print(f"[INFO] Saved {savefig}")
        summary.setdefault("plots", []).append("summary.png")

        # ---- Save individual subplots ----
        individual_plots = {
            "mse": ax_mse,
            "roc": ax_roc,
            "mass": ax_mass,
            "pt": ax_pt,
        }
        expand_left_frac, expand_right_frac = 0.12, 0.05
        expand_bottom_frac, expand_top_frac = 0.11, 0.01

        for name, ax in individual_plots.items():
            fig_local = ax.figure
            extent = ax.get_window_extent().transformed(fig_local.dpi_scale_trans.inverted())
            width, height = extent.width, extent.height
            new_extent = Bbox.from_bounds(
                extent.x0 - width * expand_left_frac,
                extent.y0 - height * expand_bottom_frac,
                width + width * (expand_left_frac + expand_right_frac),
                height + height * (expand_bottom_frac + expand_top_frac),
            )
            savefig = os.path.join(savedir, "plots", f"{name}.png")
            fig_local.savefig(savefig, dpi=200, bbox_inches=new_extent)
            print(f"[INFO] Saved {savefig}")
            summary["plots"].append(f"{name}.png")

    except Exception as e:
        warnings.warn(f"Failed to create combined summary figure: {e}")

# -------------------------
# Main evaluation function
# -------------------------

def run_full_evaluation(
    checkpoint_path: str,
    model_name: str,
    config_path: str = "data/dataset_config_small.json",
    device: str = "cpu",
    batch_size: int = 2048,
    max_jets: int = 20000,
    pt_cut=None,
    wnae_params: dict = None,
    generate_all_plots: bool = True,
    savedir: str = None
) -> Dict[str, Any]:
    """
    Run the full evaluation chain. Returns a summary dict containing computed metrics and saved paths.
    """
    # Resolve device
    device_t = torch.device(device)

    # Load model configuration
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Model name {model_name} not found in MODEL_REGISTRY.")
    model_config = MODEL_REGISTRY[model_name]
    INPUT_DIM = model_config["input_dim"]
    SAVEDIR = savedir or model_config["savedir"]
    BKG_NAME = model_config["process"]

    if wnae_params is None:
        WNAE_PARAMS = WNAE_PARAM_PRESETS["DEFAULT_WNAE_PARAMS"]
    else:
        WNAE_PARAMS = wnae_params

    os.makedirs(os.path.join(SAVEDIR, "plots"), exist_ok=True)

    # Load dataset config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Background dataset and loader
    bkg_path = config[BKG_NAME]["path"]
    bkg_dataset = load_dataset(bkg_path, input_dim=INPUT_DIM, max_jets=max_jets, pt_cut=pt_cut)
    bkg_loader = DataLoader(bkg_dataset, batch_size=batch_size, sampler=SequentialSampler(bkg_dataset))

    # Signals
    signal_loaders = {}
    for name, sample in config.items():
        if name == BKG_NAME:
            continue
        sig_dataset = load_dataset(sample["path"], input_dim=INPUT_DIM, max_jets=max_jets, pt_cut=pt_cut)
        signal_loaders[name] = DataLoader(sig_dataset, batch_size=batch_size, sampler=SequentialSampler(sig_dataset))

    # Instantiate model and load checkpoint
    model = WNAE(encoder=model_config["encoder"](), decoder=model_config["decoder"](), **WNAE_PARAMS)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device_t)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device_t)["model_state_dict"])
    model.to(device_t)
    model.eval()

    if "buffer" in checkpoint:
        print("[INFO] Loading replay buffer from checkpoint")
        stored_buffer = checkpoint["buffer"]
        try:
            if model.buffer.max_samples != len(stored_buffer):
                warnings.warn(f"Stored buffer len ({len(stored_buffer)}) different from declared buffer size {model.buffer.max_samples}. Truncating/using prefix.")
                model.buffer.buffer = stored_buffer[:model.buffer.max_samples]
            else:
                model.buffer.buffer = stored_buffer
        except Exception as e:
            warnings.warn(f"Failed to set model.buffer from checkpoint: {e}")
    else:
        warnings.warn("No replay buffer found in checkpoint.")

    savedir_plots = os.path.join(SAVEDIR, "plots")

    summary = {"savedir": SAVEDIR, "plots": [], "aucs": {}}

    if generate_all_plots:
        # Energy map with MCMC
        try:
            plot_reco_map_with_mcmc(model, savedir=savedir_plots, feature_idx=(0, 1), n_chains_to_plot=5, value_range=(-3, 3), device=device_t)
            summary["plots"].append("reco_energy_map_mcmc.png")
        except Exception as e:
            warnings.warn(f"plot_reco_map_with_mcmc failed: {e}")

        # Energy distributions
        try:
            plot_energy_distributions(model, bkg_loader, savedir=savedir_plots, device=device_t)
            summary["plots"].append("energy_distributions.png")
        except Exception as e:
            warnings.warn(f"plot_energy_distributions failed: {e}")

        # Sample vs reconstruction
        try:
            plot_sample_vs_reconstruction(model, bkg_loader, savedir=savedir_plots, device=device_t)
            summary["plots"].append("sample_reconstruction.png")
            summary["plots"].append("sample_latent.png")
        except Exception as e:
            warnings.warn(f"plot_sample_vs_reconstruction failed: {e}")

        # Checkpoint energies
        try:
            plot_checkpoint_energies(checkpoint, plot_dir=savedir_plots)
            summary["plots"].append("energies_per_batch.png")
        except Exception as e:
            warnings.warn(f"plot_checkpoint_energies failed: {e}")

    # Compute MSEs
    print("[INFO] Computing background mse...")
    bkg_mses = compute_mse(model, bkg_loader, device=device_t)
    sig_mses_dict = {}
    for name, loader in signal_loaders.items():
        print(f"[INFO] Computing mse for signal: {name}")
        sig_mses_dict[name] = compute_mse(model, loader, device=device_t)

    make_summary_plots(
        bkg_mses=bkg_mses,
        sig_mses_dict=sig_mses_dict,
        bkg_dataset=bkg_dataset,
        signal_loaders=signal_loaders,
        summary=summary,
        savedir=SAVEDIR,
        bkg_name=BKG_NAME,
    )

    # Eff vs pt (wp=0.1)
    try:
        plot_eff_vs_pt(bkg_mses, sig_mses_dict, bkg_dataset, signal_loaders, wp=0.1, savedir=os.path.join(SAVEDIR, "plots"), bkg_name=BKG_NAME)
        summary["plots"].append("eff_vs_pt_wp_0.1.png")
    except Exception as e:
        warnings.warn(f"plot_eff_vs_pt failed: {e}")

    print("[INFO] Evaluation complete.")
    return summary

# -------------------------
# CLI
# -------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a WNAE checkpoint and produce diagnostic plots.")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint (.pth) file")
    parser.add_argument("--model-name", "-m", required=True, help="Model name (must exist in MODEL_REGISTRY)")
    parser.add_argument("--config", "-C", default="data/dataset_config_small.json", help="Dataset config JSON")
    parser.add_argument("--device", "-d", default="cpu", help="Device string (e.g. cpu or cuda:0)")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-jets", type=int, default=20000)
    parser.add_argument("--pt-cut", type=float, default=None)
    parser.add_argument("--no-plots", action="store_true", help="If set, skip plot generation (only compute mses/aucs)")
    parser.add_argument("--savedir", default=None, help="Optional override for model_config['savedir']")
    parser.add_argument("--wnae-params", type=str, default=None, help="Name of WNAE parameter set from model_config")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    if args.wnae_params is not None:
        if args.wnae_params not in WNAE_PARAM_PRESETS:
            raise KeyError(f"WNAE param set '{args.wnae_params}' not found in WNAE_PARAM_PRESETS.")
        wnae_params_dict = WNAE_PARAM_PRESETS[args.wnae_params]
    else:
        wnae_params_dict = None  # run_full_evaluation will handle default


    summary = run_full_evaluation(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        config_path=args.config,
        device=args.device,
        batch_size=args.batch_size,
        max_jets=args.max_jets,
        pt_cut=args.pt_cut,
        generate_all_plots=not args.no_plots,
        savedir=args.savedir,
        wnae_params=wnae_params_dict
    )
    print(json.dumps(summary, indent=2))
