import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from utils.jet_dataset import JetDataset
from wnae import WNAE
from model_registry import MODEL_REGISTRY
import os, random
from utils.plotting_helpers import ensure_dir, plot_epoch_1d, plot_epoch_2d
import itertools
import json
# ------------------- Config ------------------- #

MODEL_NAME = "deep_ttbar"
model_config = MODEL_REGISTRY[MODEL_NAME]

DATA_PATH = json.load(open("dataset_config.json"))[model_config["process"]]["path"]
INPUT_DIM = model_config["input_dim"]
SAVEDIR = model_config["savedir"]
CHECKPOINT_PATH = f"{SAVEDIR}/wnae_checkpoint_{INPUT_DIM}.pth"
PLOT_DIR = f"{SAVEDIR}/plots/"
BATCH_SIZE = 4096
NUM_SAMPLES = 2 ** 16
LEARNING_RATE = 1e-3
N_EPOCHS = 150

#For plotting
PLOT_DISTRIBUTIONS = True
PLOT_EPOCHS  = [50]  # Final epoch is always added automatically
BINS         = np.linspace(-5.0, 5.0, 101)
N_1D_SAMPLES = 10   # how many random features to plot for non-final epochs
N_2D_SAMPLES = 5    # how many 2D scatter plots to print
RNG_SEED     = 0

WNAE_PARAMS = {
    "sampling": "pcd",
    "n_steps": 10,
    "step_size": None,
    "noise": 0.2,
    "temperature": 0.05,
    "bounds": (-4.,4.),
    "mh": False,
    "initial_distribution": "gaussian",
    "replay": True,
    "replay_ratio": 0.95,
    "buffer_size": 10000,
}
DEVICE = torch.device("cpu")
# -------------------  ------------------- #

def run_training(model, optimizer, loss_function, n_epochs, training_loader, validation_loader,
                 start_epoch=0, training_losses=None, validation_losses=None, checkpoint_prefix=None):

    training_losses = training_losses or []
    validation_losses = validation_losses or []
    for i_epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()
        training_loss = 0
        n_batches = 0

        bar_format = f"Epoch {i_epoch+1}/{start_epoch + n_epochs}: " \
                     + "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

        for batch in tqdm(training_loader, bar_format=bar_format):
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()

            if loss_function == "wnae":
                loss, train_dict = model.train_step(x)
            elif loss_function == "nae":
                loss, train_dict = model.train_step_nae(x)
            elif loss_function == "ae":
                loss, train_dict = model.train_step_ae(x, run_mcmc=True, mcmc_replay=True)

            loss.backward()
            optimizer.step()
            training_loss += train_dict["loss"]
            n_batches += 1

        training_losses.append(training_loss / n_batches)

        # Validation
        model.eval()
        validation_loss = 0
        n_batches = 0

        for batch in validation_loader:
            x = batch[0].to(DEVICE)

            if loss_function == "wnae":
                val_dict = model.validation_step(x)
            elif loss_function == "nae":
                val_dict = model.validation_step_nae(x)
            elif loss_function == "ae":
                val_dict = model.validation_step_ae(x, run_mcmc=True)
            validation_loss += val_dict["loss"]

            if(n_batches==0 and PLOT_DISTRIBUTIONS==True and (i_epoch+1 in PLOT_EPOCHS)):
                #Plotting features, positive and negative samples, only for first batch
                mcmc = val_dict["mcmc_data"]["samples"][-1].detach().cpu().numpy()
                data = x.detach().cpu().numpy()
                ep_dir = ensure_dir(os.path.join(PLOT_DIR, f"epoch_{i_epoch+1}"))

                nfeat = data.shape[1]
                if ((i_epoch +1) == (start_epoch + n_epochs)):
                    features = range(nfeat)  #plot all features at final epoch
                else:
                    features = random.Random(RNG_SEED).sample(range(nfeat), N_1D_SAMPLES)
                
                pairs = random.Random(RNG_SEED).sample(list(itertools.combinations(range(nfeat), 2)),N_2D_SAMPLES)#N_2D_SAMPLES pairs of features to plot
                # 1D fixed-binning histograms
                plot_epoch_1d(data, mcmc, ep_dir, i_epoch+1, features, BINS)
                # a couple of 2D scatters for shape sanity, fix that later
                plot_epoch_2d(data, mcmc, ep_dir, i_epoch+1, pairs, BINS)
            
            n_batches += 1

        validation_losses.append(validation_loss / n_batches)

        # Save checkpoint after each epoch
        if checkpoint_prefix:
            torch.save({
                "epoch": i_epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_losses": training_losses,
                "validation_losses": validation_losses,
            }, f"{SAVEDIR}/{checkpoint_prefix}_epoch{i_epoch + 1}.pth")

    return training_losses, validation_losses

def plot_losses(training_losses, validation_losses, save_dir):
    epochs = list(range(len(training_losses)))
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, "training_loss_plot.png")
    plt.figure()
    plt.plot(epochs, training_losses, label="Training", color="red", linewidth=2)
    plt.plot(epochs, validation_losses, label="Validation", color="blue", linestyle="dashed", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    dataset = JetDataset(DATA_PATH)

    # Split
    indices = np.arange(len(dataset))
    np.random.seed(0)
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = JetDataset(DATA_PATH, indices=train_idx, input_dim=INPUT_DIM)
    val_dataset = JetDataset(DATA_PATH, indices=val_idx, input_dim=INPUT_DIM)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(train_dataset, replacement=True, num_samples=NUM_SAMPLES))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(val_dataset, replacement=True, num_samples=NUM_SAMPLES))

    model = WNAE(
        encoder=model_config["encoder"](),
        decoder=model_config["decoder"](),
        **WNAE_PARAMS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        training_losses = checkpoint["training_losses"]
        validation_losses = checkpoint["validation_losses"]
        print(f"Loaded checkpoint from epoch {start_epoch}")
    except FileNotFoundError:
        print(f"No checkpoint found at {CHECKPOINT_PATH}. Starting training from scratch.")
        start_epoch = 0
        training_losses = []
        validation_losses = []

    # Train
    training_losses, validation_losses = run_training(
        model=model,
        optimizer=optimizer,
        loss_function="wnae",
        n_epochs=N_EPOCHS,
        training_loader=train_loader,
        validation_loader=val_loader,
        start_epoch=start_epoch,
        training_losses=training_losses,
        validation_losses=validation_losses,
        #checkpoint_prefix="training"
    )

    # Plot
    plot_losses(training_losses, validation_losses, PLOT_DIR)

    # Save checkpoint
    torch.save({
        "epoch": start_epoch + N_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_losses": training_losses,
        "validation_losses": validation_losses,
    }, CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
