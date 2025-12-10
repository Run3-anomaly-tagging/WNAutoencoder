"""
Sliced Wasserstein distance study on Gaussian distributions.

Lightweight version using POT library's sliced_wasserstein_distance.
Useful for high-dimensional tests where exact Wasserstein is too expensive.
"""
import os
import sys
import torch
import numpy as np
import json
from datetime import datetime
import uuid
from tqdm import tqdm
import ot

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

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

OUTPUT_FILE = "results/distance_study/sws_results.json"


def sample_A(n):
    return torch.randn(n, DIM) * A_STD + A_MEAN

def sample_B(n):
    return torch.randn(n, DIM) * B_STD + B_MEAN

def compute_self_distance(sample_fn, name):
    sws_vals = []
    for _ in tqdm(range(N_REPEATS), desc=f"{name}-{name} distance"):
        b1 = sample_fn(BATCH)
        b2 = sample_fn(BATCH)
        dist = ot.sliced_wasserstein_distance(b1, b2, n_projections=50, seed=0)
        sws_vals.append(dist)

    print(f"{name}-{name}:")
    print(f"   SWS: mean={np.mean(sws_vals):.3f}, std={np.std(sws_vals):.3f}")
    return sws_vals

def compute_cross_distance(sampleA_fn, sampleB_fn):
    sws_vals = []
    for _ in tqdm(range(N_REPEATS), desc="A-B distance"):
        bA = sampleA_fn(BATCH)
        bB = sampleB_fn(BATCH)
        sws = ot.sliced_wasserstein_distance(bA, bB, n_projections=50, seed=0)
        sws_vals.append(sws)

    print(f"A-B:")
    print(f"   SWS: mean={np.mean(sws_vals):.3f}, std={np.std(sws_vals):.3f}")
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
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test sliced Wasserstein distance on Gaussian distributions",
        epilog="Example: python distance_studies/sliced_wasserstein_study.py --dim 256 --batch 4096"
    )
    parser.add_argument("--batch", type=int, default=4096, help="Batch size")
    parser.add_argument("--dim", type=int, default=256, help="Feature dimension")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repetitions")
    parser.add_argument("--a-mean", type=float, default=0.0, help="Mean of distribution A")
    parser.add_argument("--a-std", type=float, default=1.0, help="Std of distribution A")
    parser.add_argument("--b-mean", type=float, default=0.0, help="Mean of distribution B")
    parser.add_argument("--b-std", type=float, default=1.0, help="Std of distribution B")
    parser.add_argument("--output", type=str, default="results/distance_study/sws_results.json", help="Output JSON file")
    args = parser.parse_args()

    BATCH = args.batch
    DIM = args.dim
    N_REPEATS = args.repeats
    A_MEAN = args.a_mean
    A_STD = args.a_std
    B_MEAN = args.b_mean
    B_STD = args.b_std
    OUTPUT_FILE = os.path.join(project_root, args.output)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("Settings:")
    print(f"  A ~ N({A_MEAN}, {A_STD}²)")
    print(f"  B ~ N({B_MEAN}, {B_STD}²)")
    print(f"  BATCH={BATCH}, DIM={DIM}, N_REPEATS={N_REPEATS}")
    print()

    sws_AA_vals = compute_self_distance(sample_A, "A")
    sws_BB_vals = compute_self_distance(sample_B, "B")
    sws_AB_vals = compute_cross_distance(sample_A, sample_B)

    print()
    print("Summary:")
    print(f"  SWS(A,A): {np.mean(sws_AA_vals):.4f} ± {np.std(sws_AA_vals):.4f}")
    print(f"  SWS(B,B): {np.mean(sws_BB_vals):.4f} ± {np.std(sws_BB_vals):.4f}")
    print(f"  SWS(A,B): {np.mean(sws_AB_vals):.4f} ± {np.std(sws_AB_vals):.4f}")

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
                "sws_vals": sws_AA_vals,
                "sws_mean": float(np.mean(sws_AA_vals)),
                "sws_std": float(np.std(sws_AA_vals))
            },
            "BB": {
                "sws_vals": sws_BB_vals,
                "sws_mean": float(np.mean(sws_BB_vals)),
                "sws_std": float(np.std(sws_BB_vals))
            },
            "AB": {
                "sws_vals": sws_AB_vals,
                "sws_mean": float(np.mean(sws_AB_vals)),
                "sws_std": float(np.std(sws_AB_vals))
            },
        }
    }

    save_json(OUTPUT_FILE, data)
    print(f"\n[INFO] Results saved under key '{key}' in: {OUTPUT_FILE}")
