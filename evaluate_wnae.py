import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, SequentialSampler

from utils.jet_dataset import JetDataset
from wnae import WNAE
from model_registry import MODEL_REGISTRY

# --- Configuration ---
CONFIG_PATH = "dataset_config.json"
BATCH_SIZE = 512
MODEL_NAME = "deep"
model_config = MODEL_REGISTRY[MODEL_NAME]
INPUT_DIM = model_config["input_dim"]
SAVEDIR = model_config["savedir"]
CHECKPOINT_PATH = f"{SAVEDIR}/wnae_checkpoint_{INPUT_DIM}.pth"
MAX_JETS = 10000
DEVICE = torch.device("cpu")

WNAE_PARAMS = {
    "sampling": "pcd",
    "n_steps": 10,
    "step_size": None,
    "noise": 0.2,
    "temperature": 0.05,
    "bounds": (-3., 3.),
    "mh": False,
    "initial_distribution": "gaussian",
    "replay": True,
    "replay_ratio": 0.95,
    "buffer_size": 10000,
}

os.makedirs(f"{SAVEDIR}/plots", exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

def load_dataset(file_path, key="Jets", max_jets=10000):
    dataset = JetDataset(file_path, input_dim=INPUT_DIM, key=key)
    indices = np.random.choice(len(dataset), size=min(len(dataset), max_jets), replace=False)
    return JetDataset(file_path, input_dim=INPUT_DIM, indices=indices)


bkg_path = config["qcd_samples"]["QCD_merged"]["path"]
bkg_dataset = load_dataset(bkg_path, max_jets=MAX_JETS)
bkg_loader = DataLoader(bkg_dataset, batch_size=BATCH_SIZE, sampler=SequentialSampler(bkg_dataset))

signal_loaders = {}
for name, sample in config["signal_samples"].items():
    sig_dataset = load_dataset(sample["path"], max_jets=MAX_JETS)
    signal_loaders[name] = DataLoader(sig_dataset, batch_size=BATCH_SIZE, sampler=SequentialSampler(sig_dataset))

model = WNAE(encoder=model_config["encoder"](),decoder=model_config["decoder"](),**WNAE_PARAMS)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model_state_dict"])
model.to(DEVICE)
model.eval()

def compute_losses(dataloader):
    losses = []
    for batch in dataloader:
        x = batch[0].to(DEVICE)
        recon_x = model.decoder(model.encoder(x))
        per_sample_loss = torch.mean((x - recon_x) ** 2, dim=1)
        losses.extend(per_sample_loss.detach().cpu().numpy())
    return np.array(losses)

print("[INFO] Computing background losses...")
bkg_losses = compute_losses(bkg_loader)

sig_losses_dict = {}
for name, loader in signal_loaders.items():
    print(f"[INFO] Computing losses for signal: {name}")
    sig_losses_dict[name] = compute_losses(loader)
# --- Combined figure: loss, ROC, mass, pt ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 1) Loss distributions ---
ax_loss = axes[0, 0]
bins_loss = np.linspace(0, 500, 101)
ax_loss.hist(bkg_losses, bins=bins_loss, histtype='step', label="QCD", density=True)
for name, losses in sig_losses_dict.items():
    ax_loss.hist(losses, bins=bins_loss, histtype='step', label=name, density=True)
ax_loss.set_xlabel("Reconstruction MSE")
ax_loss.set_ylabel("Density")
ax_loss.set_xlim([0, 500])
ax_loss.legend()

# --- 2) ROC curves ---
ax_roc = axes[0, 1]
all_labels = np.zeros_like(bkg_losses)
for name, sig_losses in sig_losses_dict.items():
    labels = np.concatenate([all_labels, np.ones_like(sig_losses)])
    scores = np.concatenate([bkg_losses, sig_losses])
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
ax_mass.hist(bkg_dataset.get_mass(), bins=bins_mass, histtype='step', density=True, label="QCD")
for name, loader in signal_loaders.items():
    sig_ds = loader.dataset
    ax_mass.hist(sig_ds.get_mass(), bins=bins_mass, histtype='step', density=True, label=name)
ax_mass.set_xlabel("Jet mass [GeV]")
ax_mass.set_ylabel("Density")
ax_mass.legend()

# --- 4) Jet pt distributions ---
ax_pt = axes[1, 1]
bins_pt = np.linspace(150, 800, 65)
ax_pt.hist(bkg_dataset.get_pt(), bins=bins_pt, histtype='step', density=True, label="QCD")
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

# --- Also save individual subplots ---
individual_plots = {
    "loss": ax_loss,
    "roc": ax_roc,
    "mass": ax_mass,
    "pt": ax_pt,
}

for name, ax in individual_plots.items():
    fig_ind = plt.figure()
    new_ax = fig_ind.add_subplot(111)
    for artist in ax.get_children():
        try:
            new_ax.add_artist(artist)
        except Exception:
            pass
    new_ax.set_xlim(ax.get_xlim())
    new_ax.set_ylim(ax.get_ylim())
    new_ax.set_xlabel(ax.get_xlabel())
    new_ax.set_ylabel(ax.get_ylabel())
    new_ax.legend(*ax.get_legend_handles_labels())
    new_ax.grid(True, alpha=0.3)
    savefig = f"{SAVEDIR}/plots/{name}.png"
    plt.savefig(savefig, dpi=200)
    plt.close(fig_ind)
    print(f"Saved {savefig}")

print("[INFO] Evaluation complete.")
