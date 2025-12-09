import torch
import numpy as np
import json
import os
from datetime import datetime
import uuid
from wnae import WNAE
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

# --------------------------
# Settings
# --------------------------
BATCH = 4096
DIM = 4
N_REPEATS = 10
A_MEAN = 0.0
A_STD  = 1.0

B_MEAN = 0.0
B_STD  = 0.5

OUTPUT_FILE = "distance_results.json"


def sample_A(n):
    return torch.randn(n, DIM) * A_STD + A_MEAN

def sample_B(n):
    return torch.randn(n, DIM) * B_STD + B_MEAN

def compute_self_distance(sample_fn, name):
    wd_vals = []
    sd_vals = []
    for _ in tqdm(range(N_REPEATS), desc=f"{name}-{name} distance"):
        b1 = sample_fn(BATCH)
        b2 = sample_fn(BATCH)
        wd = WNAE.compute_wasserstein_distance(b1, b2).item()
        sd = WNAE.compute_sinkhorn_divergence(b1, b2).item()
        wd_vals.append(wd)
        sd_vals.append(sd)

    print(f"{name}-{name}:")
    print(f"   W-dist: mean={np.mean(wd_vals):.3f}, std={np.std(wd_vals):.3f}")
    print(f"   Sinkhorn: mean={np.mean(sd_vals):.3f}, std={np.std(sd_vals):.3f}")

    return wd_vals, sd_vals


def compute_cross_distance(sampleA_fn, sampleB_fn):
    wd_vals = []
    sd_vals = []
    for _ in tqdm(range(N_REPEATS), desc="A-B distance"):
        bA = sampleA_fn(BATCH)
        bB = sampleB_fn(BATCH)
        wd = WNAE.compute_wasserstein_distance(bA, bB).item()
        sd = WNAE.compute_sinkhorn_divergence(bA, bB).item()
        wd_vals.append(wd)
        sd_vals.append(sd)

    print(f"A-B:")
    print(f"   W-dist: mean={np.mean(wd_vals):.3f}, std={np.std(wd_vals):.3f}")
    print(f"   Sinkhorn: mean={np.mean(sd_vals):.3f}, std={np.std(sd_vals):.3f}")

    return wd_vals, sd_vals

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

    wd_AA_vals, sd_AA_vals = compute_self_distance(sample_A, "A")
    wd_BB_vals, sd_BB_vals = compute_self_distance(sample_B, "B")
    wd_AB_vals, sd_AB_vals = compute_cross_distance(sample_A, sample_B)

    print(f"Ratio W(A,B)/W(A,A): {np.mean(wd_AB_vals) / np.mean(wd_AA_vals):.4f}")
    print(f"Ratio Sinkhorn(A,B)/Sinkhorn(A,A): {np.mean(sd_AB_vals) / np.mean(sd_AA_vals):.4f}")
    print()
    print(f"Difference W(A,B) - W(A,A): {np.mean(wd_AB_vals) - np.mean(wd_AA_vals):.4f}")
    print(f"Difference Sinkhorn(A,B) - Sinkhorn(A,A): {np.mean(sd_AB_vals) - np.mean(sd_AA_vals):.4f}")

    data = load_json(OUTPUT_FILE)
    key = autokey()

    data[key] = {
        "settings": {
            "A_MEAN": A_MEAN,
            "A_STD": A_STD,
            "B_MEAN": B_MEAN,
            "B_STD": B_STD,
            "BATCH": BATCH,
            "DIM": DIM,
            "N_REPEATS": N_REPEATS,
        },

        "results": {
            "AA": {
                "wd_vals": wd_AA_vals,
                "sd_vals": sd_AA_vals,
                "wd_mean": float(np.mean(wd_AA_vals)),
                "wd_std": float(np.std(wd_AA_vals)),
                "sd_mean": float(np.mean(sd_AA_vals)),
                "sd_std": float(np.std(sd_AA_vals)),
            },
            "BB": {
                "wd_vals": wd_BB_vals,
                "sd_vals": sd_BB_vals,
                "wd_mean": float(np.mean(wd_BB_vals)),
                "wd_std": float(np.std(wd_BB_vals)),
                "sd_mean": float(np.mean(sd_BB_vals)),
                "sd_std": float(np.std(sd_BB_vals)),
            },
            "AB": {
                "wd_vals": wd_AB_vals,
                "sd_vals": sd_AB_vals,
                "wd_mean": float(np.mean(wd_AB_vals)),
                "wd_std": float(np.std(wd_AB_vals)),
                "sd_mean": float(np.mean(sd_AB_vals)),
                "sd_std": float(np.std(sd_AB_vals)),
            },
        }
    }

    save_json(OUTPUT_FILE, data)
    print(f"\nSaved results under key: {key} in {OUTPUT_FILE}")
