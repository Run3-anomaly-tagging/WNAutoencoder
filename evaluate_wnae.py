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

# --- Configuration ---
CONFIG_PATH = "data/dataset_config_small.json"
#CONFIG_PATH = "data/dataset_config_alt.json"
BATCH_SIZE = 512
MODEL_NAME = "feat4_encoder32_deep_bqq"
model_config = MODEL_REGISTRY[MODEL_NAME]
INPUT_DIM = model_config["input_dim"]
SAVEDIR = model_config["savedir"]
CHECKPOINT_PATH = f"{SAVEDIR}/wnae_checkpoint_{INPUT_DIM}.pth"
MAX_JETS = 20000
DEVICE = torch.device("cpu")

#Plotting options
#PT_CUT = 300
PT_CUT = None
BKG_NAME = model_config["process"]

WNAE_PARAMS = {
    "sampling": "pcd",
    "n_steps": 10,
    "step_size": 0.1,
    "noise": None,
    "temperature": 0.05,
    "bounds": (-6.,6.),
    "mh": False,
    "initial_distribution": "gaussian",
    "replay": True,
    "replay_ratio": 0.95,
    "buffer_size": 10000,
}

os.makedirs(f"{SAVEDIR}/plots", exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

def load_dataset(file_path, key="Jets", max_jets=10000, pt_cut=None):
    tmp_ds = JetDataset(file_path, input_dim=INPUT_DIM, key=key, pt_cut=pt_cut)
    # Sample random indices from the already cut dataset
    if len(tmp_ds) > max_jets:
        sampled = np.random.choice(tmp_ds.indices, size=max_jets, replace=False)
        tmp_ds.indices = sampled
    return tmp_ds

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
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model_state_dict"])
model.to(DEVICE)
model.eval()

def compute_mse(dataloader):
    mses = []
    for batch in dataloader:
        x = batch[0].to(DEVICE)
        recon_x = model.decoder(model.encoder(x))
        per_sample_mse = torch.mean((x - recon_x) ** 2, dim=1)
        mses.extend(per_sample_mse.detach().cpu().numpy())
    return np.array(mses)

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

def plot_eff_vs_pt(bkg_mses, sig_mses_dict, bkg_dataset, signal_loaders, wp=0.1, savedir=SAVEDIR):
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

plot_eff_vs_pt(bkg_mses, sig_mses_dict, bkg_dataset, signal_loaders, wp=0.1, savedir=SAVEDIR)
#plot_eff_vs_pt(bkg_mses, sig_mses_dict, bkg_dataset, signal_loaders, wp=0.01, savedir=SAVEDIR)
#plot_eff_vs_pt(bkg_mses, sig_mses_dict, bkg_dataset, signal_loaders, wp=0.001, savedir=SAVEDIR)
print("[INFO] Evaluation complete.")
