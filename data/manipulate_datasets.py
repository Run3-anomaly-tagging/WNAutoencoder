import h5py
import numpy as np
import os
import json
from tqdm import tqdm
import subprocess

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

if __name__ == "__main__":
    
    with open("dataset_qcd_ht.json", "r") as f:
        dataset_config = json.load(f)
    #merge_qcd_ht_bins(dataset_config, "QCD_merged.h5", key="Jets")

    config_path = "../dataset_config.json"
    with open(config_path, "r") as f:
        dataset_config = json.load(f)

    qcd_path = dataset_config["QCD"]["path"]
    tt_path = dataset_config["TTto4Q"]["path"]
    
    
    #inspect_h5("/uscms/home/roguljic/nobackup/AnomalyTagging/el9/WNAutoencoder/data/svj.h5")
    #inspect_h5("/uscms/home/roguljic/nobackup/AnomalyTagging/el9/WNAutoencoder/data/TTto4Q.h5")
    #inspect_h5("/uscms/home/roguljic/nobackup/AnomalyTagging/el9/WNAutoencoder/data/TTto4Q_scaled.h5")
    split_tt_by_category(input_fp=tt_path, key="Jets", category_field="category")
    
    #merge_qcd_tt(qcd_fp=qcd_path, tt_fp=tt_path, output_fp=os.path.join(os.path.dirname(qcd_path), "qcd_tt_mixed.h5"), key="Jets", ratio=1.0)
