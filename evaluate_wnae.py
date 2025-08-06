import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import roc_curve, auc

from utils.jet_dataset import JetDataset
from wnae import WNAE
from train_shallow import Encoder, Decoder #This is important because it defines the models!
import h5py

# --- Configuration ---
FILEPATH = "/uscms_data/d3/roguljic/AnomalyTagging/el9/AutoencoderTraining/data/auc_qcd_H_signal_scaled.h5"
SAVEDIR = "shallow"
BATCH_SIZE = 512
INPUT_DIM = 256
CHECKPOINT_PATH = f"{SAVEDIR}/wnae_checkpoint_{INPUT_DIM}.pth"
MAX_JETS = 10000
DEVICE = torch.device("cpu")
WNAE_PARAMS = {
    "sampling": "pcd",
    "n_steps": 10,
    "step_size": None,
    "noise": 0.2,
    "temperature": 0.05,
    "bounds": (-3.,3.),
    "mh": False,
    "initial_distribution": "gaussian",
    "replay": True,
    "replay_ratio": 0.95,
    "buffer_size": 10000,
}

bkg_dataset = JetDataset(FILEPATH, input_dim=INPUT_DIM, key="Jets_Bkg")
sig_dataset = JetDataset(FILEPATH, input_dim=INPUT_DIM, key="Jets_Signal")

n_bkg = min(len(bkg_dataset), MAX_JETS)
n_sig = min(len(sig_dataset), MAX_JETS)

bkg_indices = np.random.choice(len(bkg_dataset), size=n_bkg, replace=False)
sig_indices = np.random.choice(len(sig_dataset), size=n_sig, replace=False)

bkg_dataset = JetDataset(FILEPATH, input_dim=INPUT_DIM, key="Jets_Bkg", indices=bkg_indices)
sig_dataset = JetDataset(FILEPATH, input_dim=INPUT_DIM, key="Jets_Signal", indices=sig_indices)

bkg_loader = DataLoader(bkg_dataset, batch_size=BATCH_SIZE, sampler=SequentialSampler(bkg_dataset))
sig_loader = DataLoader(sig_dataset, batch_size=BATCH_SIZE, sampler=SequentialSampler(sig_dataset))

# --- Initialize model ---
model = WNAE(
        encoder=Encoder(INPUT_DIM),
        decoder=Decoder(INPUT_DIM),
        **WNAE_PARAMS
    )
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model_state_dict"])
model.to(DEVICE)
model.eval()

def compute_losses(dataloader):
    losses = []
    for batch in dataloader:
        x = batch[0].to(DEVICE)
        recon_x = model.encoder(x)
        recon_x = model.decoder(recon_x)  
        per_sample_loss = torch.mean((x - recon_x) ** 2, dim=1)
        losses.extend(per_sample_loss.detach().cpu().numpy())  
    return np.array(losses)



# --- Compute losses ---
bkg_losses = compute_losses(bkg_loader)
sig_losses = compute_losses(sig_loader)

print(len(bkg_losses))
print(len(sig_losses))

# --- Plot loss distributions ---
plt.figure()
range_min, range_max = 0, 1000 
plt.hist(bkg_losses, bins=50, histtype='step',  label="QCD (background)", density=True, range=(range_min, range_max))
plt.hist(sig_losses, bins=50, histtype='step',   label="H->bb (signal)", density=True, range=(range_min, range_max))
plt.xlabel("Reconstruction MSE")
plt.ylabel("Density")
plt.xlim(range_min, range_max)
plt.legend()
plt.savefig(f"{SAVEDIR}/plots/loss_distributions.png")
plt.close()

# --- ROC Curve ---
all_losses = np.concatenate([bkg_losses, sig_losses])
all_labels = np.concatenate([np.zeros_like(bkg_losses), np.ones_like(sig_losses)])
fpr, tpr, _ = roc_curve(all_labels, all_losses)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", color="darkorange", lw=2)
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(f"{SAVEDIR}/plots/roc_curve.png")
plt.close()

print(f"[INFO] Evaluation complete. AUC: {roc_auc:.4f}")
