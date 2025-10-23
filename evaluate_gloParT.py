import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.transforms import Bbox
from utils.jet_dataset import JetDataset
import h5py

# --- Configuration ---
CONFIG_PATH = "dataset_config.json"
MAX_JETS = 20000
PT_CUT = None

# Which discriminator to use
DISC_METHOD = "get_gloParT_QCD"   #get_gloParT_QCD / get_gloParT_Tbqq / get_gloParT_Tbq
BKG_NAME = "QCD" 
SAVEDIR = "models/gloParT_QCD"

os.makedirs(SAVEDIR, exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

def load_dataset(file_path, key="Jets", max_jets=10000, pt_cut=None):
    # #For debugging
    # with h5py.File(file_path, "r") as f:
    #     if key not in f:
    #         raise ValueError(f"Dataset '{key}' not found in {file_path}")
    #     jets = f[key]
    #     print(f"[INFO] Loading {file_path}")
    #     print(f"        Available fields in '{key}': {jets.dtype.names}")

    ds = JetDataset(file_path, key=key, pt_cut=pt_cut)
    if len(ds) > max_jets:
        sampled = np.random.choice(ds.indices, size=max_jets, replace=False)
        ds.indices = sampled
    return ds

# --- Load datasets ---
bkg_dataset = load_dataset(config[BKG_NAME]["path"], max_jets=MAX_JETS, pt_cut=PT_CUT)

signal_datasets = {}
for name, sample in config.items():
    if name == BKG_NAME:
        continue
    sig_dataset = load_dataset(sample["path"], max_jets=MAX_JETS, pt_cut=PT_CUT)
    signal_datasets[name] = sig_dataset

# --- Extract discriminator values using dataset methods ---
bkg_scores = getattr(bkg_dataset, DISC_METHOD)()
bkg_scores = 1.0-bkg_scores

sig_scores_dict = {}
for name, sig_dataset in signal_datasets.items():
    sig_scores_dict[name] = getattr(sig_dataset, DISC_METHOD)()
    sig_scores_dict[name] = 1.0-sig_scores_dict[name]

# --- Combined figure: scores, ROC, mass, pt ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) Discriminator distributions
ax_disc = axes[0, 0]
x_max = 1.
bins_disc = np.linspace(0, x_max, 101)

ax_disc.hist(bkg_scores, bins=bins_disc, histtype="step", label=BKG_NAME, density=True)
for name, scores in sig_scores_dict.items():
    ax_disc.hist(scores, bins=bins_disc, histtype="step", label=name, density=True)
ax_disc.set_xlabel(f"1.-{DISC_METHOD} score")
ax_disc.set_ylabel("Density")
ax_disc.set_xlim([0, x_max])
ax_disc.legend()

# 2) ROC curves
ax_roc = axes[0, 1]
all_labels = np.zeros_like(bkg_scores)
for name, sig_scores in sig_scores_dict.items():
    labels = np.concatenate([all_labels, np.ones_like(sig_scores)])
    scores = np.concatenate([bkg_scores, sig_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
ax_roc.set_xlabel("Background mistag rate")
ax_roc.set_ylabel("Signal efficiency")
ax_roc.legend(loc="lower right")
ax_roc.grid(True, alpha=0.3)

# 3) Jet mass distributions
ax_mass = axes[1, 0]
bins_mass = np.linspace(0, 200, 101)
ax_mass.hist(bkg_dataset.get_mass(), bins=bins_mass, histtype="step", density=True, label=BKG_NAME)
for name, sig_dataset in signal_datasets.items():
    ax_mass.hist(sig_dataset.get_mass(), bins=bins_mass, histtype="step", density=True, label=name)
ax_mass.set_xlabel("Jet mass [GeV]")
ax_mass.set_ylabel("Density")
ax_mass.legend()

# 4) Jet pt distributions
ax_pt = axes[1, 1]
bins_pt = np.linspace(150, 800, 65)
ax_pt.hist(bkg_dataset.get_pt(), bins=bins_pt, histtype="step", density=True, label=BKG_NAME)
for name, sig_dataset in signal_datasets.items():
    ax_pt.hist(sig_dataset.get_pt(), bins=bins_pt, histtype="step", density=True, label=name)
ax_pt.set_xlabel("Jet $p_T$ [GeV]")
ax_pt.set_ylabel("Density")
ax_pt.set_yscale("log")
ax_pt.legend()

plt.tight_layout()
savefig = f"{SAVEDIR}/summary.png"
plt.savefig(savefig, dpi=200)
plt.close()
print(f"Saved {savefig}")
