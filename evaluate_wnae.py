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
MODEL_NAME = "shallow"
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

# --- Loss Ditributions ---
plt.figure()
bins = np.linspace(0, 1500, 151)
plt.hist(bkg_losses, bins=bins, histtype='step', label="QCD (background)", density=True)
for name, losses in sig_losses_dict.items():
    plt.hist(losses, bins=bins, histtype='step', label=name, density=True)
plt.xlabel("Reconstruction MSE")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Loss Distributions")
savefig = f"{SAVEDIR}/plots/loss_distributions_multi_signal.png"
plt.savefig(savefig)
print(f"Saved {savefig}")
plt.close()

# --- ROC Curves ---
plt.figure()
all_labels = np.zeros_like(bkg_losses)
for name, sig_losses in sig_losses_dict.items():
    labels = np.concatenate([all_labels, np.ones_like(sig_losses)])
    scores = np.concatenate([bkg_losses, sig_losses])
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: QCD vs Multiple Signals")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
savefig = f"{SAVEDIR}/plots/roc_curve_multi_signal.png"
plt.savefig(savefig)
print(f"Saved {savefig}")
plt.close()

print("[INFO] Evaluation complete.")
