import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from utils.jet_dataset import JetDataset
from wnae import WNAE

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

def plot_losses(training_losses, validation_losses, save_path):
    epochs = list(range(len(training_losses)))
    plt.figure()
    plt.plot(epochs, training_losses, label="Training", color="red", linewidth=2)
    plt.plot(epochs, validation_losses, label="Validation", color="blue", linestyle="dashed", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# ------------------- Config ------------------- #

DATA_PATH = "/uscms/home/roguljic/nobackup/AnomalyTagging/el9/AutoencoderTraining/data/merged_qcd_train_scaled.h5"
INPUT_DIM = 256
SAVEDIR = "shallow"
CHECKPOINT_PATH = f"{SAVEDIR}/wnae_checkpoint_{INPUT_DIM}.pth"
PLOT_PATH = f"{SAVEDIR}/plots/training_loss_plot.png"

BATCH_SIZE = 512
NUM_SAMPLES = 2 ** 15
LEARNING_RATE = 1e-3
N_EPOCHS = 20

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
DEVICE = torch.device("cpu")


# ------------------- Model ------------------- #

class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = input_size*2
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        #self.layer3 = nn.Linear(hidden_size, hidden_size)
        #self.layer4 = nn.Linear(hidden_size, hidden_size)
        #self.layer5 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        #x = torch.relu(self.layer3(x))
        #x = torch.relu(self.layer4(x))
        #x = torch.relu(self.layer5(x))
        return x

class Decoder(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        hidden_size = output_size*2
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
       # self.layer3 = nn.Linear(hidden_size, output_size)
       # self.layer4 = nn.Linear(hidden_size, output_size)
       # self.layer5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

# ------------------- Main ------------------- #

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
        encoder=Encoder(INPUT_DIM),
        decoder=Decoder(INPUT_DIM),
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
        checkpoint_prefix="training"
    )

    # Plot
    plot_losses(training_losses, validation_losses, PLOT_PATH)

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
