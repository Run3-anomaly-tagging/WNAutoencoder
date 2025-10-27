import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, SequentialSampler
from matplotlib.transforms import Bbox
from utils.jet_dataset import JetDataset
from wnae import WNAE
from model_config.model_registry import MODEL_REGISTRY
from model_config.model_config import DEFAULT_WNAE_PARAMS, TUTORIAL_WNAE_PARAMS

import os
import matplotlib.pyplot as plt
import numpy as np

def compute_mse(dataloader):
    mses = []
    for batch in dataloader:
        x = batch[0].to(DEVICE)
        recon_x = model.decoder(model.encoder(x))
        per_sample_mse = torch.mean((x - recon_x) ** 2, dim=1)
        mses.extend(per_sample_mse.detach().cpu().numpy())
    return np.array(mses)

def plot_checkpoint_energies(checkpoint, plot_dir="plots"):
    """
    Plot positive and negative energies per batch from a loaded checkpoint.

    Args:
        checkpoint: dictionary returned from torch.load(checkpoint_path)
        plot_dir: directory to save plots
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    pos_energies = checkpoint.get("batch_pos_energies", None)
    neg_energies = checkpoint.get("batch_neg_energies", None)
    
    if pos_energies is None or neg_energies is None:
        print("[WARNING] Positive/Negative energies not found in checkpoint.")
        print("Checkpoint keys:", checkpoint.keys())
        return
    
    pos_energies = np.array(pos_energies)
    neg_energies = np.array(neg_energies)
    
    # Plot energies per batch
    plt.figure(figsize=(8,5))
    plt.plot(pos_energies, label="Positive Energy")
    plt.plot(neg_energies, label="Negative Energy")
    plt.xlabel("Batch number")
    plt.ylabel("Energy")
    plt.legend(frameon=False)
    #plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "energies_per_batch.png")
    plt.savefig(plot_path)
    plt.savefig(plot_path.replace(".png",".pdf"))
    plt.close()
    
    print(f"[INFO] Energy plot saved to: {plot_path}")

def load_dataset(file_path, key="Jets", max_jets=10000, pt_cut=None):
    tmp_ds = JetDataset(file_path, input_dim=INPUT_DIM, key=key, pt_cut=pt_cut)
    # Sample random indices from the already cut dataset
    if len(tmp_ds) > max_jets:
        sampled = np.random.choice(tmp_ds.indices, size=max_jets, replace=False)
        tmp_ds.indices = sampled
    return tmp_ds

def plot_eff_vs_pt(bkg_mses, sig_mses_dict, bkg_dataset, signal_loaders, wp=0.1, savedir="plots"):
    """
    Plot efficiency vs jet pT for a fixed working point defined by a background mistag rate.
    
    Args:
        bkg_mses (np.ndarray): background MSE scores
        sig_mses_dict (dict): {name: np.ndarray} of signal MSE scores
        bkg_dataset (JetDataset): background dataset (provides pT)
        signal_loaders (dict): {name: DataLoader} for signals (to get dataset pT)
        wp (float): background mistag working point (e.g. 0.1 for 10%)
        savedir (str): where to save plot
    """
    # threshold from background: WP corresponds to (1 - wp) quantile
    threshold = np.percentile(bkg_mses, 100 * (1 - wp))

    bins_pt = np.linspace(150, 800, 50)  # adjust as needed
    bin_centers = 0.5 * (bins_pt[:-1] + bins_pt[1:])

    # background efficiency per pT bin
    bkg_pts = bkg_dataset.get_pt()
    bkg_eff_pt = []
    for i in range(len(bins_pt) - 1):
        mask = (bkg_pts >= bins_pt[i]) & (bkg_pts < bins_pt[i+1])
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
            mask = (sig_pts >= bins_pt[i]) & (sig_pts < bins_pt[i+1])
            if np.sum(mask) > 0:
                eff = np.mean(sig_mses[mask] > threshold)
            else:
                eff = np.nan
            sig_eff.append(eff)
        sig_eff_pt_dict[name] = sig_eff

    # plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_centers, bkg_eff_pt, label=f"{BKG_NAME} mistag (WP={wp*100:.0f}%)", linestyle="--")
    for name, eff in sig_eff_pt_dict.items():
        ax.plot(bin_centers, eff, label=f"{name}")
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0, 1.3)
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig = f"{savedir}/plots/eff_vs_pt_wp_{wp}.png"
    plt.savefig(savefig, dpi=200)
    plt.close()
    print(f"Saved {savefig}")

def plot_sample_vs_reconstruction(model, bkg_loader, savedir, device=torch.device("cpu")):
    """
    Plot one MCMC jet and one validation jet with original and reconstructed features.

    Args:
        model: trained WNAE model (should be in eval mode)
        bkg_loader: DataLoader for background/validation jets
        savedir: directory to save the plot
    """
    # Get one validation jet
    val_batch = next(iter(bkg_loader))
    val_jet = val_batch[0][0:1].to(device)  # first jet in batch

    val_energy, val_z, val_reco = model._WNAE__energy_with_samples(val_jet)

    # Get one MCMC jet
    if len(model.buffer.buffer) == 0:
        raise ValueError("MCMC buffer is empty!")
    mcmc_jet = model.buffer.buffer[0].unsqueeze(0).to(device)  # first mcmc jet in batch
    mcmc_energy, mcmc_z, mcmc_reco = model._WNAE__energy_with_samples(mcmc_jet)

    n_features = val_jet.shape[1]
    features = range(n_features)

    plt.figure(figsize=(10,5))

    plt.plot(features, val_jet[0].cpu().numpy(), 'o-', color='C0', label='Val jet')
    plt.plot(features, val_reco[0].detach().cpu().numpy(), 's--', color='C0', label='Val jet reco.')

    plt.plot(features, mcmc_jet[0].cpu().numpy(), 'o-', color='C1', label='MCMC jet')
    plt.plot(features, mcmc_reco[0].detach().cpu().numpy(), 's--', color='C1', label='MCMC jet reco.')

    plt.xlabel("Feature index")
    plt.ylabel("Feature value")

    plt.text(0.95, 0.95, f"Val energy: {val_energy.item():.1f}", transform=plt.gca().transAxes,
            horizontalalignment='right', verticalalignment='top', color='C0')
    plt.text(0.95, 0.90, f"MCMC energy: {mcmc_energy.item():.1f}", transform=plt.gca().transAxes,
            horizontalalignment='right', verticalalignment='top', color='C1')

    plt.legend()
    plt.tight_layout()
    save_path = f"{savedir}/sample_reconstruction.png"
    plt.savefig(save_path)
    plt.close()
    plt.cla()
    plt.clf()
    print(f"Saved {save_path}")

    #Latent plot
    plt.figure(figsize=(10,5))

    n_latent = val_z.shape[1]
    latent_idx = range(n_latent)

    plt.plot(latent_idx, val_z[0].detach().cpu().numpy(), 'o-', color='C0', label='Val jet z')
    plt.plot(latent_idx, mcmc_z[0].detach().cpu().numpy(), 'o-', color='C1', label='MCMC jet z')

    plt.xlabel("Latent dimension index")
    plt.ylabel("Latent value")

    plt.legend()
    plt.tight_layout()
    save_path_z = f"{savedir}/sample_latent.png"
    plt.savefig(save_path_z)
    plt.close()
    print(f"Saved {save_path_z}")

def plot_energy_distributions(model, bkg_loader, n_samples=10000, savedir="plots", device=torch.device("cpu")):
    """
    Plot distributions of positive (data) and negative (MCMC) reconstruction energies
    on separate subplots.

    Args:
        model: trained WNAE model (should be in eval mode)
        bkg_loader: DataLoader for background/validation jets
        n_samples: number of validation jets to sample
        savedir: directory to save the plot
        device: torch device
    """
    model.eval()
    E_pos_list = []
    E_neg_list = []

    # Collect positive (data) energies
    count = 0
    for batch in bkg_loader:
        jets = batch[0].to(device)
        for jet in jets:
            energy, _, _ = model._WNAE__energy_with_samples(jet.unsqueeze(0))
            E_pos_list.append(energy.item())
            count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break

    # Collect negative (MCMC) energies
    if len(model.buffer.buffer) == 0:
        raise ValueError("MCMC buffer is empty!")
    for i in range(min(n_samples, len(model.buffer.buffer))):
        mcmc_jet = model.buffer.buffer[i].unsqueeze(0).to(device)
        energy, _, _ = model._WNAE__energy_with_samples(mcmc_jet)
        E_neg_list.append(energy.item())

    # Plot distributions in two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12,5), sharey=True)

    _, x_pos_max = np.percentile(E_pos_list, [0, 99])
    _, x_neg_max = np.percentile(E_neg_list, [0, 99])

    bins_pos = np.linspace(0, x_pos_max, 50)
    bins_neg = np.linspace(0, x_neg_max, 50)

    axs[0].hist(E_pos_list, bins=bins_pos,histtype='step', color='C0', fill=False)
    axs[0].set_title("E+ (data)")
    axs[0].set_xlabel("Reconstruction energy")
    axs[0].set_ylabel("Counts")

    axs[1].hist(E_neg_list, bins=bins_neg,histtype='step', color='C1', fill=False)
    axs[1].set_title("E- (MCMC)")
    axs[1].set_xlabel("Reconstruction energy")

    # Compute 99th percentile x limits
    x_pos_max = torch.tensor(E_pos_list).kthvalue(int(0.99*len(E_pos_list)))[0].item()
    x_neg_max = torch.tensor(E_neg_list).kthvalue(int(0.99*len(E_neg_list)))[0].item()
    axs[0].set_xlim(0,right=x_pos_max)
    axs[1].set_xlim(0,right=x_neg_max)

    plt.tight_layout()
    save_path = f"{savedir}/energy_distributions.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

# --- Configuration ---
CONFIG_PATH = "data/dataset_config_small.json"
#CONFIG_PATH = "data/dataset_config_alt.json"
BATCH_SIZE = 2048
MODEL_NAME = "shallow16_encoder128_qcd"
model_config = MODEL_REGISTRY[MODEL_NAME]
INPUT_DIM = model_config["input_dim"]
SAVEDIR = model_config["savedir"] + "_sinkhorn"
CHECKPOINT_PATH = f"{SAVEDIR}/wnae_checkpoint_{INPUT_DIM}.pth"
MAX_JETS = 20000
DEVICE = torch.device("cpu")

#Plotting options
#PT_CUT = 300
PT_CUT = None
BKG_NAME = model_config["process"]

WNAE_PARAMS = TUTORIAL_WNAE_PARAMS
WNAE_PARAMS["distance"] = "sinkhorn"

os.makedirs(f"{SAVEDIR}/plots", exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

bkg_path = config[BKG_NAME]["path"]
bkg_dataset = load_dataset(bkg_path, max_jets=MAX_JETS,pt_cut=PT_CUT)
bkg_loader = DataLoader(bkg_dataset, batch_size=BATCH_SIZE, sampler=SequentialSampler(bkg_dataset))

signal_loaders = {}
for name, sample in config.items():
    if name==BKG_NAME:
        continue
    sig_dataset = load_dataset(sample["path"], max_jets=MAX_JETS,pt_cut=PT_CUT)
    signal_loaders[name] = DataLoader(sig_dataset, batch_size=BATCH_SIZE, sampler=SequentialSampler(sig_dataset))

model = WNAE(encoder=model_config["encoder"](),decoder=model_config["decoder"](),**WNAE_PARAMS)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model_state_dict"])
model.to(DEVICE)
model.eval()
if "buffer" in checkpoint:
    print("Loading replay buffer from checkpoint")
    if model.buffer.max_samples!=len(checkpoint["buffer"]):
        print(f'WARNING: stored buffer len ({len(checkpoint["buffer"])}) different from declared buffer size {model.buffer.max_samples}')
        model.buffer.buffer = checkpoint["buffer"][:model.buffer.max_samples]
    else:
        model.buffer.buffer = checkpoint["buffer"]

plot_energy_distributions(model, bkg_loader, savedir=f"{SAVEDIR}/plots")
plot_sample_vs_reconstruction(model, bkg_loader, savedir=f"{SAVEDIR}/plots")
plot_checkpoint_energies(checkpoint, plot_dir=f"{SAVEDIR}/plots/")

print("[INFO] Computing background mse...")
bkg_mses = compute_mse(bkg_loader)

sig_mses_dict = {}
for name, loader in signal_loaders.items():
    print(f"[INFO] Computing mse for signal: {name}")
    sig_mses_dict[name] = compute_mse(loader)
# --- Combined figure: mse, ROC, mass, pt ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 1) Mse distributions ---
ax_mse = axes[0, 0]

all_mses = np.concatenate([bkg_mses] + list(sig_mses_dict.values()))
_, x_max = np.percentile(all_mses, [0, 99.])


bins_mse = np.linspace(0, x_max, 101)
ax_mse.hist(bkg_mses, bins=bins_mse, histtype='step', label=BKG_NAME, density=True)
for name, mses in sig_mses_dict.items():
    ax_mse.hist(mses, bins=bins_mse, histtype='step', label=name, density=True)
ax_mse.set_xlabel("Reconstruction MSE")
ax_mse.set_ylabel("Density")
ax_mse.set_xlim([0, x_max])
ax_mse.legend()

# --- 2) ROC curves ---
ax_roc = axes[0, 1]
all_labels = np.zeros_like(bkg_mses)
for name, sig_mses in sig_mses_dict.items():
    labels = np.concatenate([all_labels, np.ones_like(sig_mses)])
    scores = np.concatenate([bkg_mses, sig_mses])
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
ax_roc.set_xlabel("Background mistag rate")
ax_roc.set_ylabel("Signal efficiency")
ax_roc.legend(loc="lower right")
ax_roc.grid(True, alpha=0.3)

# --- 3) Jet mass distributions ---
ax_mass = axes[1, 0]
bins_mass = np.linspace(0, 200, 101)
ax_mass.hist(bkg_dataset.get_mass(), bins=bins_mass, histtype='step', density=True, label=BKG_NAME)
for name, loader in signal_loaders.items():
    sig_ds = loader.dataset
    ax_mass.hist(sig_ds.get_mass(), bins=bins_mass, histtype='step', density=True, label=name)
ax_mass.set_xlabel("Jet mass [GeV]")
ax_mass.set_ylabel("Density")
ax_mass.legend()

# --- 4) Jet pt distributions ---
ax_pt = axes[1, 1]
bins_pt = np.linspace(150, 800, 65)
ax_pt.hist(bkg_dataset.get_pt(), bins=bins_pt, histtype='step', density=True, label=BKG_NAME)
for name, loader in signal_loaders.items():
    sig_ds = loader.dataset
    ax_pt.hist(sig_ds.get_pt(), bins=bins_pt, histtype='step', density=True, label=name)
ax_pt.set_xlabel("Jet $p_T$ [GeV]")
ax_pt.set_ylabel("Density")
ax_pt.set_ylim(1e-4, 3*1e-2)
ax_pt.set_yscale("log")
ax_pt.legend()

plt.tight_layout()
savefig = f"{SAVEDIR}/plots/summary.png"
plt.savefig(savefig, dpi=200)
plt.close()
print(f"Saved {savefig}")


# --- Save each subplot individually ---
#Thank you SO: https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
individual_plots = {
    "mse": ax_mse,
    "roc": ax_roc,
    "mass": ax_mass,
    "pt": ax_pt,
}

expand_left_frac = 0.12 
expand_right_frac = 0.05
expand_bottom_frac = 0.11
expand_top_frac = 0.01


for name, ax in individual_plots.items():
    fig = ax.figure
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # Slightly expand to avoid clipping labels, legends, ticks
    width = extent.width
    height = extent.height
    

    new_extent = Bbox.from_bounds(
        extent.x0 - width * expand_left_frac,
        extent.y0 - height * expand_bottom_frac,
        width + width * (expand_left_frac + expand_right_frac),
        height + height * (expand_bottom_frac + expand_top_frac)
    )
    # Save
    savefig = f"{SAVEDIR}/plots/{name}.png"
    fig.savefig(savefig, dpi=200, bbox_inches=new_extent)
    print(f"Saved {savefig}")


plot_eff_vs_pt(bkg_mses, sig_mses_dict, bkg_dataset, signal_loaders, wp=0.1, savedir=SAVEDIR)
print("[INFO] Evaluation complete.")
