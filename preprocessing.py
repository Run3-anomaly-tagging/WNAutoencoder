import h5py
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from utils.jet_dataset import JetDataset


def plot_feature_comparisons_bkg_vs_sig(h5_file, output_dir, key_bkg="Jets_Bkg", key_sig="Jets_Signal"):
    def plot_feature_comparison(bkg_vals, sig_vals, output_path, title=""):
        plt.figure(figsize=(4, 3))
        bins = 50
        value_range = (-5, 5)

        plt.hist(bkg_vals, bins=bins, color='skyblue', edgecolor='black', alpha=0.6, range=value_range, label="Background")
        plt.hist(sig_vals, bins=bins, color='salmon', edgecolor='black', alpha=0.6, range=value_range, label="Signal")

        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right", fontsize="x-small")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        print(f"Saved {output_path}")
        plt.close()

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_file, "r") as f:
        jets_bkg = f[key_bkg]
        jets_sig = f[key_sig]

        bkg_hid = jets_bkg["hidNeurons"]
        sig_hid = jets_sig["hidNeurons"]

        n_features = bkg_hid.shape[1]

        for i in range(n_features):
            bkg_vals = bkg_hid[:, i]
            sig_vals = sig_hid[:, i]
            output_path = os.path.join(output_dir, f"feature_{i}_comparison.png")
            title = f"Feature {i} Distribution"
            plot_feature_comparison(bkg_vals, sig_vals, output_path, title=title)



def plot_feature(feature_vals,output_path,title="",percentiles=[]):
    plt.figure(figsize=(4, 3))
    plt.hist(feature_vals, bins=50, color='lightgreen', edgecolor='black', range=(-5, 5))
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    legend_text=""
    for i, percentile in enumerate(percentiles):
        boundary=3+i     
        legend_text += f"% within [-{boundary},{boundary}]: {percentile:.1f}%\n"
    plt.legend([legend_text], loc="upper right", fontsize="x-small")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()



def compute_stats_on_sample(h5file, dataset_name='Jets', sample_size=10000):
    dset = h5file[dataset_name]
    n = len(dset)
    sample_size = min(sample_size, n)
    indices = np.random.choice(n, sample_size, replace=False)
    
    sample_hid = np.array([dset[i]['hidNeurons'] for i in indices])
    means = sample_hid.mean(axis=0)
    stds = sample_hid.std(axis=0)
    return means, stds

def scale_and_save(input_fp, output_fp, batch_size=1000):
    with h5py.File(input_fp, 'r') as fin:
        jets_in = fin['Jets']
        n_jets = len(jets_in)
        
        # Compute scaling stats on a sample
        means, stds = compute_stats_on_sample(fin, 'Jets', sample_size)
        
        with h5py.File(output_fp, 'w') as fout:
            dtype = jets_in.dtype
            jets_out = fout.create_dataset('Jets', shape=(n_jets,), dtype=dtype)
            
            fout.create_dataset('hidNeurons_means', data=means)
            fout.create_dataset('hidNeurons_stds', data=stds)

            for start in tqdm(range(0, n_jets, batch_size), total=(n_jets + batch_size - 1)//batch_size):
                end = min(start + batch_size, n_jets)
                batch = jets_in[start:end]
                
                batch_scaled = np.empty(batch.shape, dtype=dtype)
                for name in dtype.names:
                    if name == 'hidNeurons':
                        # Scale hidNeurons only
                        batch_scaled[name] = (batch[name] - means) / stds
                    else:
                        #Won't save jet images
                        continue
                
                jets_out[start:end] = batch_scaled

    print(f"Scaling complete. Output saved to {output_fp}")

def sanity_check_scaled_features(filepath, hist_dir="scaled_feature_histograms"):
    print(f"Running sanity check on scaled features from {filepath}")
    
    os.makedirs(hist_dir, exist_ok=True)
    
    dataset = JetDataset(filepath)
    sample_indices = np.random.choice(len(dataset), size=10000, replace=False)
    sample = torch.stack([dataset[i][0] for i in sample_indices])
    
    means = sample.mean(dim=0).numpy()
    stds = sample.std(dim=0).numpy()
    mins = sample.min(dim=0).values.numpy()
    maxs = sample.max(dim=0).values.numpy()
    
    print(f"{'Feature':>8} | {'Mean':>10} | {'StdDev':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 60)
    for i in range(sample.shape[1]):
        feature_vals = sample[:, i].numpy()

        pct_within_3 = 100.0 * np.mean(np.abs(feature_vals) <= 3)
        pct_within_4 = 100.0 * np.mean(np.abs(feature_vals) <= 4)
        pct_within_5 = 100.0 * np.mean(np.abs(feature_vals) <= 5)

        print(f"{i:>8} | {means[i]:>10.4f} | {stds[i]:>10.4f} | {mins[i]:>10.4f} | {maxs[i]:>10.4f}")
        print(f"    % within [-3,3]: {pct_within_3:.2f}%, [-4,4]: {pct_within_4:.2f}%, [-5,5]: {pct_within_5:.2f}%")
        output_path = os.path.join(hist_dir, f"scaled_feature_{i:03d}.png")
        title = f"Scaled Feature {i}"
        plot_feature(feature_vals,output_path,title=title,percentiles=[pct_within_3,pct_within_4,pct_within_5])


def apply_scaling_and_save(input_fp, output_fp, mean, std, keys=("Jets_Bkg", "Jets_Signal"), batch_size=1000):
    with h5py.File(input_fp, 'r') as fin, h5py.File(output_fp, 'w') as fout:
        for key in keys:
            if key not in fin:
                print(f"[WARNING] Key '{key}' not found in input file. Skipping.")
                continue

            jets_in = fin[key]
            n_jets = len(jets_in)
            dtype = jets_in.dtype

            jets_out = fout.create_dataset(key, shape=(n_jets,), dtype=dtype)

            for start in tqdm(range(0, n_jets, batch_size), total=(n_jets + batch_size - 1) // batch_size, desc=f"Scaling {key}"):
                end = min(start + batch_size, n_jets)
                batch = jets_in[start:end]

                batch_scaled = np.empty(batch.shape, dtype=dtype)
                name = "hidNeurons"
                batch_scaled[name] = (batch[name] - mean) / std
                jets_out[start:end] = batch_scaled

    print(f"[INFO] Scaling complete. Scaled datasets saved to: {output_fp}")


if __name__ == "__main__":
    input_filepath = "/uscms/home/roguljic/nobackup/AnomalyTagging/el9/AutoencoderTraining/data/merged_qcd_train.h5"
    output_filepath = "/uscms/home/roguljic/nobackup/AnomalyTagging/el9/AutoencoderTraining/data/merged_qcd_train_scaled.h5"
    batch_size = 20000
    sample_size = 20000  # for computing mean/std

    #scale_and_save(input_filepath, output_filepath, batch_size=batch_size)
    #sanity_check_scaled_features(output_filepath)
    with h5py.File(output_filepath, "r") as f:
        mean = f["hidNeurons_means"][:]
        std = f["hidNeurons_stds"][:]

    input_filepath_eval = "/uscms_data/d3/roguljic/AnomalyTagging/el9/AutoencoderTraining/data/auc_qcd_H_signal.h5"
    output_filepath_eval = "/uscms_data/d3/roguljic/AnomalyTagging/el9/AutoencoderTraining/data/auc_qcd_H_signal_scaled.h5"
    apply_scaling_and_save(input_filepath_eval, output_filepath_eval, mean, std, keys=("Jets_Bkg", "Jets_Signal"), batch_size=1000)
    plot_feature_comparisons_bkg_vs_sig(output_filepath_eval, "scaled_feature_histograms")