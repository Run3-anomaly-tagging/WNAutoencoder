import torch
import numpy as np
import json
import os
from datetime import datetime
import uuid
from wnae import WNAE
from tqdm import tqdm
import ot
torch.manual_seed(0)
np.random.seed(0)

# --------------------------
# Settings
# --------------------------
BATCH = 4096
DIM = 256
N_REPEATS = 10
A_MEAN = 0.0
A_STD  = 1.0

B_MEAN = 0.0
B_STD  = 1.0

OUTPUT_FILE = "sws_results.json"


def sample_A(n):
    return torch.randn(n, DIM) * A_STD + A_MEAN

def sample_B(n):
    return torch.randn(n, DIM) * B_STD + B_MEAN

def compute_self_distance(sample_fn, name):
    sws_vals = []
    for _ in range(N_REPEATS):
        b1 = sample_fn(BATCH)
        b2 = sample_fn(BATCH)
        dist = ot.sliced_wasserstein_distance(b1, b2, n_projections=50, seed=0)
        sws_vals.append(dist)

    print(f"{name}-{name}:")
    print(f"   SWS-dist: mean={np.mean(sws_vals):.3f}, std={np.std(sws_vals):.3f}")

    return sws_vals

def compute_cross_distance(sampleA_fn, sampleB_fn):
    sws_vals = []
    for _ in range(N_REPEATS):
        bA = sampleA_fn(BATCH)
        bB = sampleB_fn(BATCH)
        sws = ot.sliced_wasserstein_distance(bA, bB, n_projections=50, seed=0)
        sws_vals.append(sws)

    print(f"A-B:")
    print(f"   W-dist: mean={np.mean(sws_vals):.3f}, std={np.std(sws_vals):.3f}")

    return sws_vals

def autokey():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{uuid.uuid4().hex[:6]}"


def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    print("Settings:")
    print(f"  A_MEAN = {A_MEAN}, A_STD = {A_STD}")
    print(f"  B_MEAN = {B_MEAN}, B_STD = {B_STD}")
    print(f"  BATCH  = {BATCH}, DIM = {DIM}, N_REPEATS = {N_REPEATS}")

    sws_AA_vals = compute_self_distance(sample_A, "A")
    sws_BB_vals = compute_self_distance(sample_B, "B")
    sws_AB_vals = compute_cross_distance(sample_A, sample_B)

    # data = load_json(OUTPUT_FILE)
    # key = autokey()

    # data[key] = {
    #     "settings": {
    #         "A_MEAN": A_MEAN,
    #         "A_STD": A_STD,
    #         "B_MEAN": B_MEAN,
    #         "B_STD": B_STD,
    #         "BATCH": BATCH,
    #         "DIM": DIM,
    #         "N_REPEATS": N_REPEATS,
    #     },

    #     "results": {
    #         "AA": {
    #             "sws_vals": sws_AA_vals,
    #             "sws_mean": float(np.mean(sws_AA_vals)),
    #             "sws_std": float(np.std(sws_AA_vals))
    #         },
    #         "BB": {
    #             "sws_vals": sws_BB_vals,
    #             "sws_mean": float(np.mean(sws_BB_vals)),
    #             "sws_std": float(np.std(sws_BB_vals))
    #         },
    #         "AB": {
    #             "sws_vals": sws_AB_vals,
    #             "sws_mean": float(np.mean(sws_AB_vals)),
    #             "sws_std": float(np.std(sws_AB_vals))
    #         },
    #     }
    # }

    #save_json(OUTPUT_FILE, data)
    #print(f"\nSaved results under key: {key} in {OUTPUT_FILE}")
