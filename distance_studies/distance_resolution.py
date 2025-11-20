import torch
import numpy as np
from wnae import WNAE

torch.manual_seed(0)
np.random.seed(0)

BATCH = 4096
DIM = 8
N_REPEATS = 10
A_MEAN = 0.0
A_STD  = 1.0

B_MEAN = 0.2
B_STD  = 1.0

def sample_A(n):
    """Distribution A: Normal(A_MEAN, A_STD)."""
    return torch.randn(n, DIM) * A_STD + A_MEAN

def sample_B(n):
    """Distribution B: Normal(B_MEAN, B_STD)."""
    return torch.randn(n, DIM) * B_STD + B_MEAN

def compute_self_distance(sample_fn, name):
    wd_vals = []
    sd_vals = []
    for _ in range(N_REPEATS):
        b1 = sample_fn(BATCH)
        b2 = sample_fn(BATCH)
        wd = WNAE.compute_wasserstein_distance(b1, b2).item()
        sd = WNAE.compute_sinkhorn_divergence(b1, b2).item()
        wd_vals.append(wd)
        sd_vals.append(sd)

    print(f"{name}-{name}:")
    print(f"   W-dist: mean={np.mean(wd_vals):.6f}, std={np.std(wd_vals):.6f}")
    print(f"   Sinkhorn: mean={np.mean(sd_vals):.6f}, std={np.std(sd_vals):.6f}")
    return np.mean(wd_vals), np.mean(sd_vals)

def compute_cross_distance(sampleA_fn, sampleB_fn):
    wd_vals = []
    sd_vals = []
    for _ in range(N_REPEATS):
        bA = sampleA_fn(BATCH)
        bB = sampleB_fn(BATCH)
        wd = WNAE.compute_wasserstein_distance(bA, bB).item()
        sd = WNAE.compute_sinkhorn_divergence(bA, bB).item()
        wd_vals.append(wd)
        sd_vals.append(sd)

    print(f"A-B:")
    print(f"   W-dist: mean={np.mean(wd_vals):.6f}, std={np.std(wd_vals):.6f}")
    print(f"   Sinkhorn: mean={np.mean(sd_vals):.6f}, std={np.std(sd_vals):.6f}")
    return np.mean(wd_vals), np.mean(sd_vals)

if __name__ == "__main__":
    print(f"Settings:")
    print(f"  A_MEAN = {A_MEAN}, A_STD = {A_STD}")
    print(f"  B_MEAN = {B_MEAN}, B_STD = {B_STD}")
    print(f"  BATCH  = {BATCH}, DIM = {DIM}, N_REPEATS = {N_REPEATS}")

    wd_AA, sd_AA = compute_self_distance(sample_A, "A")
    wd_BB, sd_BB = compute_self_distance(sample_B, "B")

    wd_AB, sd_AB = compute_cross_distance(sample_A, sample_B)

    print(f"Ratio W(A,B)/W(A,A): {wd_AB / wd_AA:.4f}")
    print(f"Ratio Sinkhorn(A,B)/Sinkhorn(A,A): {sd_AB / sd_AA:.4f}")
    print()
    print(f"Difference W(A,B) - W(A,A): {wd_AB - wd_AA:.4f}")
    print(f"Difference Sinkhorn(A,B) - Sinkhorn(A,A): {sd_AB - sd_AA:.4f}")
