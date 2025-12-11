"""
Diagnostic script to check if auxiliary variables are within expected [0,1] range.
Scans HDF5 files and reports any values outside the valid range.
"""

import os
import sys
import h5py
import numpy as np
import json

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def check_auxiliary_ranges(filepath, aux_keys):
    """
    Check if auxiliary variables are within [0, 1] range.
    
    Args:
        filepath: Path to HDF5 file
        aux_keys: List of auxiliary feature keys to check
    
    Returns:
        dict: Statistics for each auxiliary variable
    """
    print(f"\n{'='*80}")
    print(f"Checking: {filepath}")
    print(f"{'='*80}")
    
    with h5py.File(filepath, 'r') as f:
        jets = f['Jets']
        n_jets = len(jets)
        print(f"Total jets: {n_jets:,}")
        
        results = {}
        
        for aux_key in aux_keys:
            if aux_key not in jets.dtype.names:
                print(f"\n[ERROR] {aux_key} not found in file!")
                print(f"Available fields: {jets.dtype.names}")
                continue
            
            print(f"\n{'-'*80}")
            print(f"Checking: {aux_key}")
            print(f"{'-'*80}")
            
            # Load data
            data = jets[aux_key][:]
            
            # Compute statistics
            min_val = np.min(data)
            max_val = np.max(data)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Check for out-of-range values
            below_zero = np.sum(data < 0)
            above_one = np.sum(data > 1)
            in_range = np.sum((data >= 0) & (data <= 1))
            
            # Find extreme values
            min_idx = np.argmin(data)
            max_idx = np.argmax(data)
            
            print(f"Statistics:")
            print(f"  Min:  {min_val:.6f} (at index {min_idx})")
            print(f"  Max:  {max_val:.6f} (at index {max_idx})")
            print(f"  Mean: {mean_val:.6f}")
            print(f"  Std:  {std_val:.6f}")
            
            print(f"\nRange Check:")
            print(f"  Values < 0:       {below_zero:,} ({100*below_zero/n_jets:.4f}%)")
            print(f"  Values in [0,1]:  {in_range:,} ({100*in_range/n_jets:.4f}%)")
            print(f"  Values > 1:       {above_one:,} ({100*above_one/n_jets:.4f}%)")
            
            if below_zero > 0:
                print(f"\n[WARNING] Found {below_zero:,} values below 0!")
                # Show first few violations
                below_indices = np.where(data < 0)[0]
                n_show = min(10, len(below_indices))
                print(f"  First {n_show} violations:")
                for idx in below_indices[:n_show]:
                    print(f"    Index {idx}: {data[idx]:.6f}")
            
            if above_one > 0:
                print(f"\n[WARNING] Found {above_one:,} values above 1!")
                # Show first few violations
                above_indices = np.where(data > 1)[0]
                n_show = min(10, len(above_indices))
                print(f"  First {n_show} violations:")
                for idx in above_indices[:n_show]:
                    print(f"    Index {idx}: {data[idx]:.6f}")
            
            if below_zero == 0 and above_one == 0:
                print(f"\n[OK] All values are within [0, 1] range!")
            
            results[aux_key] = {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val,
                'below_zero': below_zero,
                'above_one': above_one,
                'in_range': in_range,
                'total': n_jets
            }
    
    return results


if __name__ == "__main__":
    # Load dataset config
    config_path = os.path.join(script_dir, "dataset_config.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract all file paths from config
    FILES = []
    for process_name, process_info in config.items():
        if "path" in process_info:
            FILES.append((process_name, process_info["path"]))
    
    # Auxiliary keys to check
    AUX_KEYS = ['globalParT3_QCD', 'globalParT3_TopbWqq']
    
    print("=" * 80)
    print("AUXILIARY VARIABLE RANGE CHECK")
    print("=" * 80)
    print(f"Loaded {len(FILES)} files from {config_path}")
    print(f"Checking files for auxiliary variables: {AUX_KEYS}")
    print(f"Expected range: [0, 1]")
    
    all_results = {}
    
    for process_name, filepath in FILES:
        full_path = os.path.join(project_root, filepath)
        if not os.path.exists(full_path):
            print(f"\n[SKIP] {process_name}: File not found: {full_path}")
            continue
        
        results = check_auxiliary_ranges(full_path, AUX_KEYS)
        all_results[process_name] = results
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for process_name, results in all_results.items():
        print(f"\n{process_name}:")
        for aux_key, stats in results.items():
            status = "OK" if stats['below_zero'] == 0 and stats['above_one'] == 0 else "VIOLATIONS"
            print(f"  {aux_key}: [{stats['min']:.4f}, {stats['max']:.4f}] - {status}")
            if status == "VIOLATIONS":
                print(f"    Out of range: {stats['below_zero'] + stats['above_one']:,} / {stats['total']:,}")
