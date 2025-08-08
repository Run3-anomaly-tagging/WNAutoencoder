#!/usr/bin/env python
# train_for_plots.py — WNAE training + fixed-binning plots (all features at last epoch)
import os, random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt

from utils.jet_dataset import JetDataset
from wnae import WNAE
from model_registry import MODEL_REGISTRY

# ------------------- Config ------------------- #
MODEL_NAME   = "deep"   # <- pick from your MODEL_REGISTRY
DATA_PATH    = "/uscms/home/roguljic/nobackup/AnomalyTagging/el9/AutoencoderTraining/data/merged_qcd_train_scaled.h5"

# training
BATCH_SIZE   = 512
NUM_SAMPLES  = 2 ** 15
LR           = 1e-3
N_EPOCHS     = 10
DEVICE       = torch.device("cpu")

# plotting
PLOT_EPOCHS  = [1, 3, 5, 10]  # final epoch is always added automatically
BINS         = np.linspace(-5.0, 5.0, 101)  # 100 bins, range [-5,5]
N_1D_SAMPLES = 6   # how many random features to plot for non-final epochs
N_2D_SAMPLES = 2   # quick 2D scatters for sanity
RNG_SEED     = 0

# WNAE specifics
WNAE_PARAMS = dict(
    sampling="pcd", n_steps=10, step_size=None, noise=0.2, temperature=0.05,
    bounds=(-3., 3.), mh=False, initial_distribution="gaussian",
    replay=True, replay_ratio=0.95, buffer_size=10000,
)
# ------------------------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def plot_losses(tr, val, png_path):
    ensure_dir(os.path.dirname(png_path))
    epochs = list(range(len(tr)))
    plt.figure()
    plt.plot(epochs, tr,  label="Training",   linewidth=2)
    plt.plot(epochs, val, label="Validation", linestyle="--", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

def plot_epoch_1d(data, mcmc, outdir, epoch, features, bins):
    ensure_dir(outdir)
    for feat in features:
        plt.hist(data[:, feat], bins=bins, histtype='step', density=True, label='data')
        plt.hist(mcmc[:, feat], bins=bins, histtype='step', density=True, label='MCMC')
        plt.legend()
        plt.title(f'Epoch {epoch} — feat {feat}')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'epoch{epoch}_feat{feat}.png'))
        plt.close()

def plot_epoch_2d(data, mcmc, outdir, epoch, rng, n_pairs=2, xlim=(-5,5), ylim=(-5,5)):
    ensure_dir(outdir)
    d = data.shape[1]
    for _ in range(min(n_pairs, max(0, d // 2))):
        x, y = rng.sample(range(d), 2)
        plt.scatter(data[:, x], data[:, y], s=2, alpha=.3, label='data')
        plt.scatter(mcmc[:, x], mcmc[:, y], s=2, alpha=.3, label='MCMC')
        plt.xlim(*xlim); plt.ylim(*ylim)
        plt.legend()
        plt.title(f'Epoch {epoch} — (feat {x}, feat {y})')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'epoch{epoch}_2d_{x}_{y}.png'))
        plt.close()

def main():
    # registry-driven
    cfg       = MODEL_REGISTRY[MODEL_NAME]
    INPUT_DIM = cfg["input_dim"]
    SAVEDIR   = ensure_dir(cfg["savedir"])
    PLOTDIR   = ensure_dir(os.path.join(SAVEDIR, "plots"))
    LOSS_PNG  = os.path.join(PLOTDIR, "training_loss_plot.png")
    CKPT_PATH = os.path.join(SAVEDIR, f"wnae_checkpoint_{INPUT_DIM}.pth")

    # seeds
    np.random.seed(RNG_SEED); random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)

    # datasets/loaders
    ds_full = JetDataset(DATA_PATH)
    idx     = np.random.permutation(len(ds_full))
    split   = int(0.8 * len(idx))
    ds_tr   = JetDataset(DATA_PATH, indices=idx[:split],  input_dim=INPUT_DIM)
    ds_val  = JetDataset(DATA_PATH, indices=idx[split:], input_dim=INPUT_DIM)

    tr_loader = DataLoader(ds_tr,  batch_size=BATCH_SIZE,
                           sampler=RandomSampler(ds_tr,  replacement=True, num_samples=NUM_SAMPLES))
    va_loader = DataLoader(ds_val, batch_size=BATCH_SIZE,
                           sampler=RandomSampler(ds_val, replacement=True, num_samples=NUM_SAMPLES))

    # model/optim
    model = WNAE(encoder=cfg["encoder"](), decoder=cfg["decoder"](), **WNAE_PARAMS).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    # maybe resume
    start_epoch, tr_losses, val_losses = 0, [], []
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except Exception as e:
            print(f"[info] model strict load failed: {e}; retrying strict=False")
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        try:
            optim.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"[warn] optimizer state not loaded: {e}")
        start_epoch = int(ckpt.get("epoch", 0))
        tr_losses   = list(ckpt.get("training_losses", []))
        val_losses  = list(ckpt.get("validation_losses", []))
        print(f"Resumed from epoch {start_epoch}")

    # ensure final epoch gets plotted (for ALL features)
    final_epoch_index = start_epoch + N_EPOCHS - 1
    plot_epochs = set(PLOT_EPOCHS)
    plot_epochs.add(final_epoch_index)
    rng = random.Random(RNG_SEED)

    # --- training loop with validation + snapshots ---
    for epoch in range(start_epoch, start_epoch + N_EPOCHS):
        # train
        model.train()
        tot_tr, n_tr = 0.0, 0
        bar = f"Epoch {epoch+1}/{start_epoch + N_EPOCHS}: " + "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        for batch in tqdm(tr_loader, bar_format=bar):
            x = batch[0].to(DEVICE)
            optim.zero_grad()
            loss, out = model.train_step(x)   # WNAE
            loss.backward()
            optim.step()
            tot_tr += float(out["loss"]); n_tr += 1
        tr_losses.append(tot_tr / max(1, n_tr))

        # validate + maybe snapshot (NO torch.no_grad(): MCMC needs autograd on x)
        model.eval()
        tot_va, n_va = 0.0, 0
        snapshot_done = False
        for i, batch in enumerate(va_loader):
            x = batch[0].to(DEVICE)
            # allow autograd so Langevin sampler can compute ∂E/∂x
            vdict = model.validation_step(x)
            tot_va += float(vdict["loss"]); n_va += 1

            # For epochs we want to plot, grab the first val batch’s data+mcmc
            if (epoch in plot_epochs) and (i == 0) and (not snapshot_done):
                try:
                    mcmc = vdict["mcmc_data"]["samples"][-1].detach().cpu().numpy()
                    data = x.detach().cpu().numpy()
                    ep_dir = ensure_dir(os.path.join(PLOTDIR, f"epoch_{epoch}"))

                    nfeat = data.shape[1]
                    if epoch == final_epoch_index:
                        features = range(nfeat)  # plot ALL features at final epoch
                    else:
                        sel = min(N_1D_SAMPLES, nfeat)
                        features = rng.sample(range(nfeat), sel)

                    # 1D fixed-binning histograms
                    plot_epoch_1d(data, mcmc, ep_dir, epoch, features, BINS)
                    # a couple of 2D scatters for shape sanity
                    plot_epoch_2d(data, mcmc, ep_dir, epoch, rng, n_pairs=N_2D_SAMPLES,
                                  xlim=(BINS[0], BINS[-1]), ylim=(BINS[0], BINS[-1]))
                    snapshot_done = True
                except Exception as e:
                    print(f"[warn] epoch {epoch}: could not snapshot MCMC/data: {e}")

        val_losses.append(tot_va / max(1, n_va))

    # save loss plot + checkpoint
    plot_losses(tr_losses, val_losses, LOSS_PNG)
    torch.save({
        "epoch": start_epoch + N_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "training_losses": tr_losses,
        "validation_losses": val_losses,
    }, CKPT_PATH)

    print(f"Saved checkpoint: {CKPT_PATH}")
    print(f"Loss plot      : {LOSS_PNG}")
    print(f"Per-epoch plots: {PLOTDIR}")

if __name__ == "__main__":
    main()
