import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import os, random, sys
import itertools
import json
import time

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir  # train.py is at project root
sys.path.insert(0, project_root)

from utils.jet_dataset import JetDataset
from wnae import WNAE
from model_config.model_registry import MODEL_REGISTRY
from model_config.model_config import WNAE_PARAM_PRESETS
from utils.plotting_helpers import ensure_dir, plot_epoch_1d, plot_epoch_2d, plot_aux_scatter
from evaluate_wnae import run_full_evaluation 

def compute_1d_projection_distance(pos_samples, neg_samples, bins=50, value_range=(-4, 4)):
    """Compute a distance between 1D histograms of positive and negative samples.

    Args:
        pos_samples: Tensor (n_samples, n_features)
        neg_samples: Tensor (n_samples, n_features)
        bins: number of bins for histograms
        value_range: tuple (min, max) for histogram bins

    Returns:
        dict of per-feature distances
    """
    if(pos_samples.shape[0]!=neg_samples.shape[0]):
        print(f"WARNING:compute_1d_projection_distance sample sizes - {pos_samples.shape[0]} and {neg_samples.shape[0]}")
        n_samples = min(pos_samples.shape[0], neg_samples.shape[0])
    else:
        n_samples = pos_samples.shape[0]
    pos = pos_samples[:n_samples].detach().cpu().numpy()
    neg = neg_samples[:n_samples].detach().cpu().numpy()

    distances = {}
    nfeat = pos.shape[1]
    for f in range(nfeat):
        h_pos, _ = np.histogram(pos[:, f], bins=bins, range=value_range, density=True)
        h_neg, _ = np.histogram(neg[:, f], bins=bins, range=value_range, density=True)
        distances[f] = np.sum(np.abs(h_pos - h_neg))
    return np.mean(list(distances.values()))

def run_training(model, optimizer, loss_function, n_epochs, training_loader, validation_loader,  plot_config, checkpoint_path=None,save_every=20, device=torch.device("cpu"),lr_factor=None, lr_patience=10, lr_min=1e-6,force_lr=None):

    start_epoch = 0
    training_losses, validation_losses = [], []
    batch_pos_energies, batch_neg_energies = [], []
    epoch_pos_energies = []
    epoch_neg_energies = []

    scheduler = None
    if lr_factor is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=lr_min)


    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if force_lr is not None:
            optimizer.param_groups[0]["lr"] = force_lr
            print(f"Forced LR = {force_lr}")
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
            elif loss_function == "cnae":
                loss, train_dict = model.train_step_cnae(x)
            elif loss_function == "ae":
                loss, train_dict = model.train_step_ae(x, run_mcmc=True, mcmc_replay=True)

            loss.backward()
            optimizer.step()

            pos_e = train_dict.get("positive_energy", 0)
            neg_e = train_dict.get("negative_energy", 0)
            batch_pos_energies.append(pos_e)
            batch_neg_energies.append(neg_e)
            epoch_pos_energy = epoch_pos_energy + pos_e
            epoch_neg_energy = epoch_neg_energy + neg_e


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
        val_pos_energy = 0
        val_neg_energy = 0
        val_proj_distance = 0
        n_batches = 0
        #with torch.no_grad():
        for batch in validation_loader:
            x = batch[0].to(device, non_blocking=True)
    
            if loss_function == "wnae":
                val_dict = model.validation_step(x)
            elif loss_function == "nae":
                val_dict = model.validation_step_nae(x)
            elif loss_function == "cnae":
                val_dict = model.validation_step_cnae(x)
            elif loss_function == "ae":
                val_dict = model.validation_step_ae(x, run_mcmc=True)
            validation_loss += val_dict["loss"]

            val_pos_energy += val_dict.get("positive_energy", 0)
            val_neg_energy += val_dict.get("negative_energy", 0)

            # Compute projection distance for monitoring (first batch only)
            pos_samples = x
            neg_samples = val_dict["mcmc_data"]["samples"][-1]
            val_proj_distance += compute_1d_projection_distance(pos_samples, neg_samples)

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

                aux_dim = plot_config.get("aux_dim", 0)
                if aux_dim >= 2:
                    plot_aux_scatter(data, mcmc, ep_dir, i_epoch+1, aux_dim=aux_dim)

            n_batches += 1
    
        val_proj_distance =  val_proj_distance / n_batches
        val_pos_energy = val_pos_energy / n_batches
        val_neg_energy = val_neg_energy / n_batches

        if scheduler is not None:
            scheduler.step(validation_loss / n_batches)            

        validation_losses.append(validation_loss / n_batches)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {i_epoch+1}/{start_epoch + n_epochs} | "
          f"Train Loss: {training_losses[-1]:.4f} | "
          f"Val Loss: {validation_losses[-1]:.4f} | "
          f"E+: {avg_pos_energy:.2f} | E-: {avg_neg_energy:.2f} | "
          f"Val E+: {val_pos_energy:.2f} | Val E-: {val_neg_energy:.2f} | "
          f"Val proj. distance: {val_proj_distance:.2f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.3e} | "
          f"Time: {epoch_time:.1f}s")

        if (start_epoch + n_epochs -1 == i_epoch):
            save_path = checkpoint_path
        elif(i_epoch%save_every==0):
            save_path = checkpoint_path.replace(".pth",f"_epoch_{i_epoch}.pth")
        else:
            save_path = None

        if save_path:
            torch.save({
                "epoch": i_epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_losses": training_losses,
                "validation_losses": validation_losses,
                "batch_pos_energies": batch_pos_energies,
                "val_proj_dist": val_proj_distance,
                "batch_neg_energies": batch_neg_energies,
                "epoch_pos_energies": epoch_pos_energies,
                "epoch_neg_energies": epoch_neg_energies,
                "buffer": model.buffer.buffer
            }, save_path)

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

    MODEL_NAME = "deep_bottleneck_qcd_bqq_aux2"
    model_config = MODEL_REGISTRY[MODEL_NAME]
    CONFIG_PATH = os.path.join(project_root, "data", "dataset_config.json")
    
    # Validate config file exists
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Dataset config not found: {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r") as f:
        dataset_config = json.load(f)
    
    # Handle single process or list of processes
    processes = model_config["process"]
    if isinstance(processes, str):
        processes = [processes]
    
    # Validate all processes exist and collect paths
    DATA_PATHS = []
    for proc in processes:
        if proc not in dataset_config:
            raise KeyError(f"Process '{proc}' not found in {CONFIG_PATH}")
        path = dataset_config[proc]["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}\nCheck dataset_config.json paths")
        DATA_PATHS.append(path)
    
    # For single file, unwrap the list
    if len(DATA_PATHS) == 1:
        DATA_PATH = DATA_PATHS[0]
    else:
        DATA_PATH = DATA_PATHS
    
    INPUT_DIM = model_config["input_dim"]
    AUX_DIM = model_config.get("aux_dim", 0)
    AUX_KEYS = ['globalParT3_QCD', 'globalParT3_TopbWqq']
    AUX_QUANTILE_TRANSFORMER = os.path.join(project_root, "data", "aux_quantile_transformer.pkl")
    
    BATCH_SIZE = 4096
    NUM_SAMPLES = 2 ** 16
    LEARNING_RATE = 1e-4
    FORCE_LR = None
    LR_PLATEAU_FACTOR = 0.8
    N_EPOCHS = 20
    LOSS_FUNCTION = "wnae"
    DISTANCE = "sliced_wasserstein"
    WNAE_PRESET = "CFG1"
    
    # Use savedir from registry + append WNAE_PRESET
    SAVEDIR = os.path.join(project_root, "models", f"{model_config['savedir']}_{WNAE_PRESET}")
    CHECKPOINT_PATH = os.path.join(SAVEDIR, "checkpoint.pth")
    PLOT_DIR = os.path.join(SAVEDIR, "plots")

    plot_config = {
        "plot_distributions": True,
        "plot_epochs": [],
        "bins": np.linspace(-5.0, 5.0, 101),
        "n_1d_samples": 2,
        "n_2d_samples": 1,
        "rng_seed": 0,
        "plot_dir": PLOT_DIR,
        "aux_dim": len(AUX_KEYS) if AUX_KEYS else 0
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WNAE_PARAMS = WNAE_PARAM_PRESETS[WNAE_PRESET].copy()
    WNAE_PARAMS["device"] = DEVICE
    WNAE_PARAMS["distance"] = DISTANCE

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Process: {model_config['process']}")
    print(f"Data: {DATA_PATH}")
    print(f"Input dim: {INPUT_DIM}")
    if AUX_KEYS:
        print(f"Auxiliary keys: {AUX_KEYS} (dim={AUX_DIM})")
        print(f"Total model input dim: {INPUT_DIM + AUX_DIM}")
    print(f"Loss function: {LOSS_FUNCTION}")
    print(f"Distance: {DISTANCE}")
    print(f"WNAE preset: {WNAE_PRESET}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    dataset = JetDataset(DATA_PATH, aux_keys=AUX_KEYS if AUX_KEYS else None)
    ensure_dir(SAVEDIR)
    
    indices = np.arange(len(dataset))
    np.random.seed(0)
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = JetDataset(
        DATA_PATH,
        indices=train_idx,
        input_dim=INPUT_DIM,
        aux_keys=AUX_KEYS if AUX_KEYS else None,
        aux_quantile_transformer=AUX_QUANTILE_TRANSFORMER if AUX_KEYS else None
    )
    val_dataset = JetDataset(
        DATA_PATH,
        indices=val_idx,
        input_dim=INPUT_DIM,
        aux_keys=AUX_KEYS if AUX_KEYS else None,
        aux_quantile_transformer=AUX_QUANTILE_TRANSFORMER if AUX_KEYS else None
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=RandomSampler(train_dataset, replacement=True, num_samples=NUM_SAMPLES),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=RandomSampler(val_dataset, replacement=True, num_samples=NUM_SAMPLES),
        pin_memory=True
    )
    
    model = WNAE(
        encoder=model_config["encoder"](),
        decoder=model_config["decoder"](),
        **WNAE_PARAMS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    training_losses, validation_losses, _, _ = run_training(
        model=model,
        optimizer=optimizer,
        loss_function=LOSS_FUNCTION,
        n_epochs=N_EPOCHS,
        training_loader=train_loader,
        validation_loader=val_loader,
        plot_config=plot_config,
        checkpoint_path=CHECKPOINT_PATH, 
        device=DEVICE,
        lr_factor=LR_PLATEAU_FACTOR,
        force_lr=FORCE_LR
    )

    plot_losses(training_losses, validation_losses, PLOT_DIR)

    print("\n[INFO] Starting full evaluation of trained checkpoint...")
    summary = None
    try:
        summary = run_full_evaluation(
            checkpoint_path=CHECKPOINT_PATH,
            model_name=MODEL_NAME,
            config_path=CONFIG_PATH,
            device=str(DEVICE),
            batch_size=BATCH_SIZE,
            wnae_params=WNAE_PARAMS,
            generate_all_plots=True,
            aux_keys=AUX_KEYS
        )
    except Exception as e:
        import traceback
        print(f"[ERROR] Full evaluation failed: {e}")
        traceback.print_exc()

    if summary is not None:
        print("\n[INFO] Evaluation summary:")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()