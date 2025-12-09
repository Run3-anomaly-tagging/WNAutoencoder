import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler
from utils.jet_dataset import JetDataset
from model_config.model_registry import MODEL_REGISTRY
from model_config.model_config import WNAE_PARAM_PRESETS
from wnae import WNAE

def sample_batch(dataset: JetDataset, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset, replacement=False), pin_memory=True)
    batch = next(iter(loader))[0]
    return batch

if __name__ == "__main__":
    model_name = "deep_bottleneck_qcd"
    checkpoints = []
    cfg = "CFG1"
    batch_size = 4096
    wnae_params = WNAE_PARAM_PRESETS[cfg]
    model_config = MODEL_REGISTRY[model_name]
    model = WNAE(encoder=model_config["encoder"](),decoder=model_config["decoder"](),**wnae_params)
    SAVEDIR = model_config["savedir"]+f"_ae_{cfg}"
    CHECKPOINT_PATH = f"../{SAVEDIR}/checkpoint.pth"
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False,map_location=torch.device('cpu'))
    #model.load_state_dict(checkpoint["model_state_dict"])
    buffer = checkpoint["buffer"]
    print(checkpoint['training_losses'][-1])
    batch1_n = buffer[:batch_size]
    batch2_n = buffer[-batch_size:]
    nn_distance = WNAE.compute_sliced_wasserstein_distance(batch1_n, batch2_n).item()


    dataset_path = "../data/QCD_merged_scaled.h5"
    dataset = JetDataset(dataset_path, input_dim=256)
    batch1 = sample_batch(dataset, batch_size=batch_size)
    batch2 = sample_batch(dataset, batch_size=batch_size)
    pp_distance = WNAE.compute_sliced_wasserstein_distance(batch1, batch2).item()
    np_distance = WNAE.compute_sliced_wasserstein_distance(batch2, batch2_n).item()
    dataset.close()

    print(f"nn distance: {nn_distance}")
    print(f"pp distance: {pp_distance}")
    print(f"np distance: {np_distance}")

