#!/usr/bin/env python
# train_for_plots.py  – layer names fixed for checkpoint loading

import os, random, torch, numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from utils.jet_dataset import JetDataset
from wnae import WNAE

# ------------------- Config ------------------- #
plot_epochs   = [0, 2, 4]
DATA_PATH     = "/uscms/home/roguljic/nobackup/AnomalyTagging/el9/AutoencoderTraining/data/merged_qcd_train_scaled.h5"
INPUT_DIM     = 256
CHECKPOINT    = f"wnae_checkpoint_{INPUT_DIM}.pth"
LOSS_PLOT_PNG = "plots/training_loss_plot.png"

BATCH_SIZE    = 256
NUM_SAMPLES   = 2 ** 15
LR            = 1e-3
N_EPOCHS      = 3
DEVICE        = torch.device("cpu")

WNAE_PARAMS = dict(
    sampling="pcd", n_steps=10, step_size=None, noise=0.2, temperature=0.05,
    bounds=(-3., 3.), mh=False, initial_distribution="gaussian",
    replay=True, replay_ratio=0.95, buffer_size=10000,
)
# ------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden = input_size * 2
        self.layer1 = nn.Linear(input_size, hidden)
        self.layer2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        hidden = output_size * 2
        self.layer1 = nn.Linear(hidden, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)   # no activation here

# ------------------------------------------------------------------ #
def run_training(model, optimizer, loss_kind, n_epochs,
                 train_loader, val_loader, plot_epochs,
                 start_epoch=0, tr_losses=None, val_losses=None,
                 mcmc_samples_list=None):
    tr_losses         = tr_losses or []
    val_losses        = val_losses or []
    mcmc_samples_list = mcmc_samples_list or []
    plot_epochs       = set(plot_epochs)

    for epoch in range(start_epoch, start_epoch + n_epochs):
        # --- training loop ---
        model.train()
        total_train_loss = 0
        n_train_batches = 0
        bar_fmt = f"Epoch {epoch+1}/{start_epoch + n_epochs}: " + "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        for batch in tqdm(train_loader, bar_format=bar_fmt):
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            if   loss_kind == "wnae":
                loss, out = model.train_step(x)
            elif loss_kind == "nae":
                loss, out = model.train_step_nae(x)
            else:
                loss, out = model.train_step_ae(x, run_mcmc=True, mcmc_replay=True)
            loss.backward()
            optimizer.step()
            total_train_loss += out["loss"]
            n_train_batches += 1
        tr_losses.append(total_train_loss / n_train_batches)

        # --- validation loop ---
        model.eval()
        total_val_loss = 0
        n_val_batches = 0
        for i_batch, batch in enumerate(val_loader):
            x = batch[0].to(DEVICE)
            if   loss_kind == "wnae":
                vdict = model.validation_step(x)
            elif loss_kind == "nae":
                vdict = model.validation_step_nae(x)
            else:
                vdict = model.validation_step_ae(x, run_mcmc=True)
            total_val_loss += vdict["loss"]
            n_val_batches += 1

            # grab one batch of data + MCMC samples
            if (epoch in plot_epochs) and (i_batch == 0):
                data_samples = x.cpu()
                mcmc_samples = vdict["mcmc_data"]["samples"][-1].cpu()
                mcmc_samples_list.append((data_samples, mcmc_samples))

        val_losses.append(total_val_loss / n_val_batches)

    return tr_losses, val_losses, mcmc_samples_list
# ------------------------------------------------------------------ #

def plot_losses(tr, val, png_path):
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    epochs = list(range(len(tr)))
    plt.figure()
    plt.plot(epochs, tr,  label="Training",   color="red")
    plt.plot(epochs, val,  label="Validation", color="blue", ls="--")
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(png_path)
    plt.close()

def quick_plots(data, mcmc, tag):
    os.makedirs("plots", exist_ok=True)
    rnd = random.Random(0)
    # four 1-D histograms
    for feat in rnd.sample(range(data.shape[1]), 4):
        plt.hist(data[:, feat], 50, histtype='step', label='data')
        plt.hist(mcmc[:, feat], 50, histtype='step', label='MCMC')
        plt.legend(); plt.title(f'{tag} — feat {feat}')
        plt.savefig(f'plots/{tag}_feat{feat}.png'); plt.close()
    # two 2-D scatters
    for _ in range(2):
        x, y = rnd.sample(range(data.shape[1]), 2)
        plt.scatter(data[:, x], data[:, y], s=2, alpha=.3, label='data')
        plt.scatter(mcmc[:, x], mcmc[:, y], s=2, alpha=.3, label='MCMC')
        plt.legend(); plt.title(f'{tag} — ({x},{y})')
        plt.savefig(f'plots/{tag}_{x}_{y}.png'); plt.close()

# ------------------- Main ------------------- #
def main():
    # prepare datasets
    full_ds  = JetDataset(DATA_PATH)
    idx      = np.random.permutation(len(full_ds))
    split    = int(0.8 * len(idx))
    train_ds = JetDataset(DATA_PATH, indices=idx[:split],  input_dim=INPUT_DIM)
    val_ds   = JetDataset(DATA_PATH, indices=idx[split:], input_dim=INPUT_DIM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=RandomSampler(train_ds, replacement=True, num_samples=NUM_SAMPLES))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              sampler=RandomSampler(val_ds,   replacement=True, num_samples=NUM_SAMPLES))

    # build model + optimizer
    model = WNAE(encoder=Encoder(INPUT_DIM), decoder=Decoder(INPUT_DIM), **WNAE_PARAMS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # optionally resume
    start_epoch = 0
    tr_losses, val_losses = [], []
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt["epoch"]
        tr_losses     = ckpt["training_losses"]
        val_losses    = ckpt["validation_losses"]
        print(f"Resumed from epoch {start_epoch}")

    # run training & collect samples
    tr_losses, val_losses, mcmc_list = run_training(
        model, optimizer, "wnae", N_EPOCHS,
        train_loader, val_loader,
        plot_epochs=plot_epochs,
        start_epoch=start_epoch,
        tr_losses=tr_losses,
        val_losses=val_losses,
        mcmc_samples_list=[],
    )

    # save loss plot + checkpoint
    plot_losses(tr_losses, val_losses, LOSS_PLOT_PNG)
    torch.save({
        "epoch": start_epoch + N_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_losses": tr_losses,
        "validation_losses": val_losses,
    }, CHECKPOINT)

    # finally, write the quick histograms & scatters
    for ep, (d, m) in zip(plot_epochs, mcmc_list):
        quick_plots(d.numpy(), m.numpy(), tag=f'epoch{ep}')

if __name__ == "__main__":
    main()
