import h5py
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import json
import subprocess
from utils.jet_dataset import JetDataset

def inspect_h5(filepath):
    with h5py.File(filepath, "r") as f:
        print(f"\nInspecting: {filepath}")
        print("=" * 60)

        first_printed = False  # flag to print only the first jet

        def visitor(name, obj):
            nonlocal first_printed
            if isinstance(obj, h5py.Dataset):
                print(f"[Dataset] {name}, shape={obj.shape}, dtype={obj.dtype}")
                if obj.dtype.names:
                    print("   Fields:", obj.dtype.names)

                    if not first_printed:
                        print("\nFirst jet content:")
                        print(obj[0])  # prints all fields of the first jet
                        first_printed = True

            elif isinstance(obj, h5py.Group):
                print(f"[Group]   {name}")

        f.visititems(visitor)

def split_tt_by_category(input_fp, key="Jets", category_field="jet_category"):
    category_names = {0: "unm", 1: "qq", 2: "bq", 3: "bqq"}
    
    with h5py.File(input_fp, "r") as fin:
        jets = fin[key]
        dtype = jets.dtype
        
        jet_cat_values = jets[category_field][:].astype(np.int32)
        categories = np.unique(jet_cat_values)
        print(f"[INFO] Found categories: {categories}")  

        base_dir = os.path.dirname(input_fp)
        base_name = os.path.basename(input_fp).replace(".h5", "")
        
        for cat in categories:
            mask = jet_cat_values == cat
            jets_cat = jets[mask]
            
            cat_name = category_names.get(cat, f"cat{cat}")
            out_fp = os.path.join(base_dir, f"{base_name}_{cat_name}.h5")
            
            with h5py.File(out_fp, "w") as fout:
                fout.create_dataset(key, data=jets_cat, dtype=dtype)
            
            print(f"[INFO] Saved {len(jets_cat)} jets to {out_fp}")

def merge_qcd_tt(qcd_fp, tt_fp, output_fp, key="Jets", ratio=1.0, batch_size=10000):
    #Ratio is the number of n_tt/n_qcd
    with h5py.File(qcd_fp, "r") as f_qcd, h5py.File(tt_fp, "r") as f_tt:
        qcd = f_qcd[key]
        tt = f_tt[key]
        n_qcd = len(qcd)
        n_tt = int(len(qcd) * ratio)
        qcd_indices = np.random.permutation(len(qcd))[:n_qcd]
        tt_indices = np.random.permutation(len(tt))[:n_tt]
        total_jets = n_qcd + n_tt
        dtype = qcd.dtype
        print(f"[INFO] Mixing {n_qcd} QCD jets with {n_tt} TT jets")

        with h5py.File(output_fp, "w") as fout:
            out_ds = fout.create_dataset(key, shape=(total_jets,), dtype=dtype)
            for start in tqdm(range(0, n_qcd, batch_size), desc="Writing QCD jets"):
                end = min(start + batch_size, n_qcd)
                idx_batch = np.sort(qcd_indices[start:end])
                out_ds[start:end] = qcd[idx_batch]
            for start in tqdm(range(0, n_tt, batch_size), desc="Writing TT jets"):
                end = min(start + batch_size, n_tt)
                idx_batch = np.sort(tt_indices[start:end])
                out_ds[n_qcd + start:n_qcd + end] = tt[idx_batch]
        print(f"[INFO] Merged dataset saved to {output_fp}")

def merge_qcd_ht_bins(config, output_fp, key="Jets"):
    redirector = "root://cmseos.fnal.gov/"
    ht_bins = [
        "QCD_HT-400to600", "QCD_HT-600to800", "QCD_HT-800to1000",
        "QCD_HT-1000to1200", "QCD_HT-1200to1500", "QCD_HT-1500to2000",
        "QCD_HT-2000toInf"
    ]

    local_files, xsecs, n_jets = {}, {}, {}

    for bin_name in ht_bins:
        info = config[bin_name]
        remote_path = redirector + info["path"]
        local_fp = os.path.basename(info["path"])
        if not os.path.exists(local_fp):
            print(f"[INFO] Copying {remote_path} → {local_fp}")
            subprocess.run(["xrdcp", "-f", remote_path, local_fp], check=True)
        else:
            print(f"[INFO] Found local copy of {local_fp}")

        local_files[bin_name] = local_fp
        xsecs[bin_name] = info["xsec"]
        with h5py.File(local_fp, "r") as f:
            n_jets[bin_name] = len(f[key])
        print(f"[INFO] {bin_name}: {n_jets[bin_name]} jets, xsec={xsecs[bin_name]}")

    ref_bin = ht_bins[0]
    ref_jets = n_jets[ref_bin]
    ref_xsec = xsecs[ref_bin]

    target_jets = {ref_bin: ref_jets}
    for bin_name in ht_bins[1:]:
        target = int(ref_jets * (xsecs[bin_name] / ref_xsec))
        target = min(target, n_jets[bin_name])
        target_jets[bin_name] = target
        print(f"[INFO] {bin_name}: keeping {target}/{n_jets[bin_name]} jets")

    total_jets = sum(target_jets.values())
    print(f"[INFO] Total jets in merged file: {total_jets}")

    os.rename(local_files[ref_bin], output_fp)
    print(f"[INFO] Using {ref_bin} as base → {output_fp}")

    with h5py.File(output_fp, "a") as fout:
        jets_ds = fout[key]
        jets_ds.resize((total_jets,))
        offset = ref_jets

        for bin_name in ht_bins[1:]:
            n_keep = target_jets[bin_name]
            if n_keep == 0:
                os.remove(local_files[bin_name])
                continue

            local_fp = local_files[bin_name]
            with h5py.File(local_fp, "r") as f:
                jets_ds[offset:offset + n_keep] = f[key][:n_keep]

            offset += n_keep
            os.remove(local_fp)
            print(f"[INFO] Appended {n_keep} jets from {bin_name} and deleted {local_fp}")

    print(f"[INFO] Merged QCD HT bins saved to {output_fp}")


def plot_feature_comparisons_bkg_vs_sig(h5_file, output_dir, key_bkg="Jets_Bkg", key_sig="Jets_Signal"):
    # Plot and save histograms comparing each feature's distribution between background and signal jets.
    # To be adapted to work with two files (this version only works with single file that contains both  bkg and signal jets)
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
    # Plot and save a histogram of a single feature's values, optionally annotating percentiles within specified ranges.
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
    # Compute mean and standard deviation of 'hidNeurons' features on a random sample from the dataset.
    dset = h5file[dataset_name]
    n = len(dset)
    sample_size = min(sample_size, n)
    indices = np.random.choice(n, sample_size, replace=False)
    
    sample_hid = np.array([dset[i]['hidNeurons'] for i in indices])
    means = sample_hid.mean(axis=0)
    stds = sample_hid.std(axis=0)
    return means, stds

def scale_and_save(input_fp, output_fp, batch_size=1000):
    # Scale 'hidNeurons' features of all jets in input file to zero mean and unit variance, then save to output file.
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
                        batch_scaled[name] = (batch[name] - means) / stds
                    elif (name =="pt" or name=="mass" or name=="category"):
                        batch_scaled[name] = batch[name]
                    else: #Skip jet images, phi, and eta
                        continue 

                
                jets_out[start:end] = batch_scaled

    print(f"Scaling complete. Output saved to {output_fp}")

def sanity_check_scaled_features(filepath, hist_dir="scaled_feature_histograms"):
    # Perform sanity checks by sampling scaled features, printing statistics, and saving feature histograms.
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
    # Apply given scaling (mean/std) to specified datasets in input file and save the scaled data to output file.
    with h5py.File(input_fp, 'r') as fin, h5py.File(output_fp, 'w') as fout:
        for key in keys:
            if key not in fin:
                print(list(fin.keys()))
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

                for name in dtype.names:
                    if name == "hidNeurons":
                        batch_scaled[name] = (batch[name] - mean) / std
                    elif name in ("pt", "mass", "category"):
                        batch_scaled[name] = batch[name]
                    else:
                        continue  # skip other fields
                jets_out[start:end] = batch_scaled

    print(f"[INFO] Scaling complete. Scaled datasets saved to: {output_fp}")


if __name__ == "__main__":

    #Switches
    merge_qcd_ht = False
    scale_merged_qcd = False
    scaling_sanity_check = False
    apply_qcd_scaling_to_signal =False
    split_tt_per_category = False
    merge_qcd_tt_flag = False


    if merge_qcd_ht:
        with open("dataset_qcd_ht.json", "r") as f:
            dataset_config = json.load(f)
        merge_qcd_ht_bins(dataset_config, "QCD_merged.h5", key="Jets")

    if scale_merged_qcd:
        input_filepath = "QCD_merged.h5"
        output_filepath = "QCD_merged_scaled.h5"
        batch_size = 20000
        sample_size = 20000 # for computing mean/std we don't need all jets

        scale_and_save(input_filepath, output_filepath, batch_size=batch_size)
        if scaling_sanity_check:
            sanity_check_scaled_features(output_filepath)

    if split_tt_per_category:
        config_path = "dataset_config.json"
        with open(config_path, "r") as f:
            dataset_config = json.load(f)

        qcd_path = dataset_config["QCD"]["path"]
        tt_path = dataset_config["TTto4Q"]["path"]
    
        #This is for debugging
        #inspect_h5("svj.h5")
        #inspect_h5("TTto4Q.h5")
        #inspect_h5("TTto4Q_scaled.h5")
    
        split_tt_by_category(input_fp=tt_path, key="Jets", category_field="category")

    if merge_qcd_tt_flag:
        merge_qcd_tt(qcd_fp=qcd_path, tt_fp=tt_path, output_fp=os.path.join(os.path.dirname(qcd_path), "qcd_tt_mixed.h5"), key="Jets", ratio=1.0)


    if  apply_qcd_scaling_to_signal:
        # Load precomputed mean and std from the scaled training file to apply the same scaling elsewhere.
        with h5py.File(output_filepath, "r") as f:
            mean = f["hidNeurons_means"][:]
            std = f["hidNeurons_stds"][:]

        input_filepath = "GluGluHto2B.h5"
        output_filepath = "GluGluHto2B_scaled.h5"
        apply_scaling_and_save(input_filepath, output_filepath, mean, std, keys=["Jets"], batch_size=1000)

        input_filepath = "svj.h5"
        output_filepath = "svj_scaled.h5"
        apply_scaling_and_save(input_filepath, output_filepath, mean, std, keys=["Jets"], batch_size=1000)

        input_filepath = "Yto4Q.h5"
        output_filepath = "Yto4Q_scaled.h5"
        apply_scaling_and_save(input_filepath, output_filepath, mean, std, keys=["Jets"], batch_size=1000)

        input_filepath = "TTto4Q.h5"
        output_filepath = "TTto4Q_scaled.h5"
        apply_scaling_and_save(input_filepath, output_filepath, mean, std, keys=["Jets"], batch_size=1000)

        input_filepath = "ZJets800.h5"
        output_filepath = "ZJets800_scaled.h5"
        apply_scaling_and_save(input_filepath, output_filepath, mean, std, keys=["Jets"], batch_size=1000)


        input_filepath = "WJets800.h5"
        output_filepath = "WJets800_scaled.h5"
        apply_scaling_and_save(input_filepath, output_filepath, mean, std, keys=["Jets"], batch_size=1000)


