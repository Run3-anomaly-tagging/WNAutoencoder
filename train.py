import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from utils.jet_dataset import JetDataset
from wnae import WNAE
from model_config.model_registry import MODEL_REGISTRY
from model_config.model_config import WNAE_PARAM_PRESETS
import os, random
from utils.plotting_helpers import ensure_dir, plot_epoch_1d, plot_epoch_2d
import itertools
import json
import time 
from evaluate_wnae import run_full_evaluation 


def run_training(model, optimizer, loss_function, n_epochs, training_loader, validation_loader,  plot_config, checkpoint_path=None,save_every=20, device=torch.device("cpu")):

    start_epoch = 0
    training_losses, validation_losses = [], []
    batch_pos_energies, batch_neg_energies = [], []
    epoch_pos_energies = []
    epoch_neg_energies = []

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        training_losses = ckpt.get("training_losses", [])
        validation_losses = ckpt.get("validation_losses", [])
        batch_pos_energies = ckpt.get("batch_pos_energies", [])
        batch_neg_energies = ckpt.get("batch_neg_energies", [])
        epoch_pos_energies = ckpt.get("epoch_pos_energies", [])
        epoch_neg_energies = ckpt.get("epoch_neg_energies", [])
        if "buffer" in ckpt:
            print("Loading replay buffer from checkpoint")
            loaded_buffer = ckpt["buffer"]
            if model.buffer.max_samples != len(loaded_buffer):
                print(f"WARNING: stored buffer len ({len(loaded_buffer)}) "
                      f"different from declared buffer size {model.buffer.max_samples}")
                loaded_buffer = loaded_buffer[:model.buffer.max_samples]

            model.buffer.buffer = [b.to(device, non_blocking=True) for b in loaded_buffer]

    plot_epochs = sorted(set(plot_config["plot_epochs"] + [start_epoch + n_epochs]))#Add the last epoch to the list for plotting
    for i_epoch in range(start_epoch, start_epoch + n_epochs):
        epoch_start_time = time.time()
        model.train()
        training_loss = 0
        n_batches = 0
        epoch_pos_energy = 0
        epoch_neg_energy = 0

        bar_format = f"Epoch {i_epoch+1}/{start_epoch + n_epochs}: " \
                     + "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

        for batch in tqdm(training_loader, bar_format=bar_format):
            x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()

            if loss_function == "wnae":
                loss, train_dict = model.train_step(x)
            elif loss_function == "nae":
                loss, train_dict = model.train_step_nae(x)
            elif loss_function == "ae":
                loss, train_dict = model.train_step_ae(x, run_mcmc=True, mcmc_replay=True)

            loss.backward()
            optimizer.step()

            pos_e = train_dict.get("positive_energy", None)
            neg_e = train_dict.get("negative_energy", None)
            batch_pos_energies.append(pos_e)
            batch_neg_energies.append(neg_e)
            epoch_pos_energy = epoch_pos_energy + pos_e
            epoch_neg_energy = epoch_neg_energy + neg_e

            #print(f"E+: {train_dict['positive_energy']:.2f}, E-: {train_dict['negative_energy']:.2f}")

            training_loss += train_dict["loss"]
            n_batches += 1

        avg_pos_energy = epoch_pos_energy / n_batches
        avg_neg_energy = epoch_neg_energy / n_batches
        training_losses.append(training_loss / n_batches)
        epoch_pos_energies.append(avg_pos_energy)
        epoch_neg_energies.append(avg_neg_energy)

        # Validation
        model.eval()
        validation_loss = 0
        n_batches = 0
        #with torch.no_grad():
        for batch in validation_loader:
            x = batch[0].to(device, non_blocking=True)
    
            if loss_function == "wnae":
                val_dict = model.validation_step(x)
            elif loss_function == "nae":
                val_dict = model.validation_step_nae(x)
            elif loss_function == "ae":
                val_dict = model.validation_step_ae(x, run_mcmc=True)
            validation_loss += val_dict["loss"]
    
            if(n_batches==0 and plot_config["plot_distributions"]==True and (i_epoch+1 in plot_epochs)):
                #Plotting features, positive and negative samples, only for first batch
                mcmc = val_dict["mcmc_data"]["samples"][-1].detach().cpu().numpy()
                data = x.detach().cpu().numpy()
                ep_dir = ensure_dir(os.path.join(plot_config["plot_dir"], f"epoch_{i_epoch+1}"))
    
                nfeat = data.shape[1]
                if ((i_epoch +1) == (start_epoch + n_epochs)):
                    features = range(nfeat)  #plot all features at final epoch
                else:
                    features = random.Random(plot_config["rng_seed"]).sample(range(nfeat), plot_config["n_1d_samples"])
                
                pairs = random.Random(plot_config["rng_seed"]).sample(list(itertools.combinations(range(nfeat), 2)),plot_config["n_2d_samples"])#N_2D_SAMPLES pairs of features to plot
                # 1D fixed-binning histograms
                plot_epoch_1d(data, mcmc, ep_dir, i_epoch+1, features, plot_config["bins"])
                # a couple of 2D scatters for shape sanity
                plot_epoch_2d(data, mcmc, ep_dir, i_epoch+1, pairs)
            
            n_batches += 1

        validation_losses.append(validation_loss / n_batches)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {i_epoch+1}/{start_epoch + n_epochs} | "
          f"Train Loss: {training_losses[-1]:.4f} | "
          f"Val Loss: {validation_losses[-1]:.4f} | "
          f"Avg E+: {avg_pos_energy:.2f} | Avg E-: {avg_neg_energy:.2f} | "
          f"Time: {epoch_time:.1f}s")

        save_epoch = i_epoch%save_every==0 or (start_epoch + n_epochs -1 == i_epoch)
        if checkpoint_path and save_epoch:
            torch.save({
                "epoch": i_epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_losses": training_losses,
                "validation_losses": validation_losses,
                "batch_pos_energies": batch_pos_energies,
                "batch_neg_energies": batch_neg_energies,
                "epoch_pos_energies": epoch_pos_energies,
                "epoch_neg_energies": epoch_neg_energies,
                "buffer": model.buffer.buffer
            }, checkpoint_path)

    return training_losses, validation_losses, epoch_pos_energies, epoch_neg_energies

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
    # ------------------- Config ------------------- #

    MODEL_NAME = "shallow16_encoder128_qcd"
    model_config = MODEL_REGISTRY[MODEL_NAME]
    CONFIG_PATH = "data/dataset_config_small.json"
    DATA_PATH = json.load(open(CONFIG_PATH))[model_config["process"]]["path"]
    #DATA_PATH = json.load(open("data/dataset_config_alt.json"))[model_config["process"]]["path"]
    INPUT_DIM = model_config["input_dim"]
    SAVEDIR = model_config["savedir"]+"_sinkhorn"
    CHECKPOINT_PATH = f"{SAVEDIR}/wnae_checkpoint_{INPUT_DIM}.pth"
    PLOT_DIR = f"{SAVEDIR}/plots/"
    BATCH_SIZE = 2048
    NUM_SAMPLES = 2 ** 16
    LEARNING_RATE = 1e-3
    N_EPOCHS = 1

    #To plot pos. and neg. distributions during training
    plot_config = {
    "plot_distributions": True,
    "plot_epochs": [100],
    "bins": np.linspace(-5.0, 5.0, 101),
    "n_1d_samples": 2,
    "n_2d_samples": 1,
    "rng_seed": 0,
    "plot_dir": PLOT_DIR
    }

    WNAE_PARAMS = WNAE_PARAM_PRESETS["TUTORIAL_WNAE_PARAMS"]
    WNAE_PARAMS["distance"] = "sinkhorn"
    DEVICE = torch.device("cpu")
    #DEVICE = torch.device("cuda")
    # -------------------  ------------------- #

    dataset = JetDataset(DATA_PATH)
    ensure_dir(SAVEDIR)

    # Split
    indices = np.arange(len(dataset))
    np.random.seed(0)
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = JetDataset(DATA_PATH, indices=train_idx, input_dim=INPUT_DIM)
    val_dataset = JetDataset(DATA_PATH, indices=val_idx, input_dim=INPUT_DIM)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(train_dataset, replacement=True, num_samples=NUM_SAMPLES),pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(val_dataset, replacement=True, num_samples=NUM_SAMPLES),pin_memory=True)
    print(f"Device is {DEVICE}")
    model = WNAE(
        encoder=model_config["encoder"](),
        decoder=model_config["decoder"](),
        **WNAE_PARAMS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    training_losses, validation_losses, _, _ = run_training(model=model,optimizer=optimizer,loss_function="wnae",n_epochs=N_EPOCHS,training_loader=train_loader,validation_loader=val_loader,checkpoint_path=CHECKPOINT_PATH, plot_config=plot_config, device=DEVICE)
    plot_losses(training_losses, validation_losses, PLOT_DIR)

    print("\n[INFO] Starting full evaluation of trained checkpoint...")
    try:
        summary = run_full_evaluation(
            checkpoint_path=CHECKPOINT_PATH,
            model_name=MODEL_NAME,
            config_path=CONFIG_PATH,
            device=str(DEVICE),
            batch_size=BATCH_SIZE,
            wnae_params=WNAE_PARAMS,
            generate_all_plots=True
        )
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"[WARN] Full evaluation failed: {e}")

if __name__ == "__main__":
    main()
