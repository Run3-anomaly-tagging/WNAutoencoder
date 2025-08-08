#!/usr/bin/env python
import h5py, numpy as np, matplotlib.pyplot as plt, random, argparse, os, mplhep as hep
hep.style.use("CMS")

def load(fname):
    with h5py.File(fname, "r") as f:
        return f["X"][:]

def main(pos_file, neg_file, outdir, n_1d=6, n_2d=3, seed=42):
    rng = random.Random(seed)
    pos, neg = load(pos_file), load(neg_file)
    ndim = pos.shape[1]
    os.makedirs(outdir, exist_ok=True)

    # ---- 1-D histograms
    one_d = rng.sample(range(ndim), n_1d)
    for i in one_d:
        plt.figure()
        plt.hist(pos[:, i], bins=100, density=True, histtype="step", label="data")
        plt.hist(neg[:, i], bins=100, density=True, histtype="step", label="MCMC")
        plt.xlabel(f"feature {i}")
        plt.ylabel("p.d.f.")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{outdir}/hist_{i}.png"); plt.close()

    # ---- 2-D scatters
    two_d = [rng.sample(range(ndim), 2) for _ in range(n_2d)]
    for i, j in two_d:
        plt.figure()
        plt.scatter(pos[:, i], pos[:, j], s=3, alpha=0.3, label="data")
        plt.scatter(neg[:, i], neg[:, j], s=3, alpha=0.3, label="MCMC")
        plt.xlabel(f"feat {i}"); plt.ylabel(f"feat {j}")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{outdir}/scatter_{i}_{j}.png"); plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pos", required=True, help="H5 with training data")
    p.add_argument("--neg", required=True, help="H5 with MCMC samples")
    p.add_argument("--outdir", default="plots")
    p.add_argument("--n1d", type=int, default=6)
    p.add_argument("--n2d", type=int, default=3)
    args = p.parse_args()
    main(args.pos, args.neg, args.outdir, args.n1d, args.n2d)
