import os
import json
import numpy as np
import torch
from utils.jet_dataset import JetDataset
from wnae import WNAE
from model_registry import MODEL_REGISTRY
import matplotlib.pyplot as plt

def load_first_n_jets(file_path, n=1, key="Jets", pt_cut=None):
    ds = JetDataset(file_path, input_dim=INPUT_DIM, key=key, pt_cut=pt_cut)
    x = torch.stack([ds[i][0] for i in range(n)])  # stack first n jets
    return x

# --- Configuration ---
N_JETS = 10000
CONFIG_PATH = "dataset_config.json"
MODEL_NAME = "feat4_encoder32_shallow_qcd"
model_config = MODEL_REGISTRY[MODEL_NAME]
INPUT_DIM = model_config["input_dim"]
SAVEDIR = model_config["savedir"]
CHECKPOINT_PATH = f"{SAVEDIR}/wnae_checkpoint_{INPUT_DIM}.pth"
DEVICE = torch.device("cpu")
WNAE_PARAMS = { #Needed for MCMC
    "sampling": "pcd",
    "n_steps": 20,
    "step_size": 0.2,
    "noise": None,
    "temperature": 0.05,
    "bounds": (-6.,6.),
    "mh": False,
    "initial_distribution": "gaussian",
    "replay": True,
    "replay_ratio": 0.5,
    "buffer_size": 10000,
}

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


qcd_path = config["QCD"]["path"]
ttbar_path = config["TTto4Q"]["path"]
qcd_jets = load_first_n_jets(qcd_path, n=N_JETS)
ttbar_jets = load_first_n_jets(ttbar_path, n=N_JETS)

# --- Eval ---
model = WNAE(encoder=model_config["encoder"](), decoder=model_config["decoder"](),**WNAE_PARAMS)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()
model._WNAE__set_shapes(qcd_jets)

#import pprint
#pprint.pprint(model.__dict__)
mcmc = model.run_mcmc(n_samples=N_JETS,replay=True)
mcmc_numpy = mcmc.detach().cpu().numpy()

with torch.no_grad():
    qcd_eval = model.evaluate(qcd_jets.to(DEVICE))
    ttbar_eval = model.evaluate(ttbar_jets.to(DEVICE))
    mcmc_eval = model.evaluate(mcmc.to(DEVICE))
    qcd_errs = qcd_eval["reco_errors"].detach().cpu().numpy()
    ttbar_errs = ttbar_eval["reco_errors"].detach().cpu().numpy()
    mcmc_errs = mcmc_eval["reco_errors"].detach().cpu().numpy()


# --- Plotting ---
n_features = mcmc_numpy.shape[1]
fig, axes = plt.subplots(1, n_features, figsize=(4*n_features, 3))
bins = np.linspace(-4, 4, 81)
for i in range(n_features):
    ax = axes[i] if n_features > 1 else axes
    ax.hist(mcmc_numpy[:, i],histtype="step", bins=bins, label="MCMC", linewidth=1.5)
    ax.hist(qcd_jets[:, i],histtype="step", bins=bins, label="QCD", linewidth=1.5)
    ax.hist(ttbar_jets[:, i],histtype="step", bins=bins, label="TTbar", linewidth=1.5) 
    ax.legend()
    ax.set_xlim(-4,4)
    ax.set_xlabel(f"Feature {i}")
    ax.set_ylabel("Count")

plt.tight_layout()
print(f"Saving {MODEL_NAME}/sanity_feat_mcmc.png")
plt.savefig(f"{MODEL_NAME}/sanity_feat_mcmc.png")

fig, ax = plt.subplots(figsize=(6,4))
xmax = np.max([np.percentile(qcd_errs, 99), np.percentile(ttbar_errs, 99)])
bins = np.linspace(0, xmax, 100)

ax.hist(qcd_errs, bins=bins, histtype="step", label="QCD", linewidth=1.5)
ax.hist(ttbar_errs, bins=bins, histtype="step", label="TTbar", linewidth=1.5)
ax.hist(mcmc_errs, bins=bins, histtype="step", label="MCMC", linewidth=1.5)
ax.set_xlim(0,xmax)
ax.set_xlabel("Reconstruction Error")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
print(f"Saving {MODEL_NAME}/sanity_mse.png")
plt.savefig(f"{MODEL_NAME}/sanity_mse.png")