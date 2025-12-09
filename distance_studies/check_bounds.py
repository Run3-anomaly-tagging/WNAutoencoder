"""
Check how many samples have at least one feature outside [-4, 4] bounds.
"""

import numpy as np
import json
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.jet_dataset import JetDataset


def check_bounds(data, lower=-4, upper=4):
    """
    Check bounds violations.
    
    Returns:
        dict with statistics
    """
    n_samples, n_dims = data.shape
    
    # Check per-sample violations
    out_of_bounds_mask = (data < lower) | (data > upper)
    samples_with_violations = np.any(out_of_bounds_mask, axis=1)
    
    # Check per-dimension violations
    dims_violations = np.sum(out_of_bounds_mask, axis=0)
    
    # Statistics
    n_violations = np.sum(out_of_bounds_mask)
    n_samples_violated = np.sum(samples_with_violations)
    pct_samples_violated = 100 * n_samples_violated / n_samples
    pct_features_violated = 100 * n_violations / (n_samples * n_dims)
    
    # Find extremes
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Count violations per sample
    violations_per_sample = np.sum(out_of_bounds_mask, axis=1)
    mean_violations = np.mean(violations_per_sample[samples_with_violations]) if n_samples_violated > 0 else 0
    
    results = {
        'n_samples': n_samples,
        'n_dims': n_dims,
        'bounds': [lower, upper],
        'n_samples_with_violations': int(n_samples_violated),
        'pct_samples_with_violations': float(pct_samples_violated),
        'n_total_violations': int(n_violations),
        'pct_features_violated': float(pct_features_violated),
        'min_value': float(min_val),
        'max_value': float(max_val),
        'mean_violations_per_violated_sample': float(mean_violations),
        'dims_with_most_violations': [(int(i), int(dims_violations[i])) for i in np.argsort(dims_violations)[-5:][::-1]]
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Check bounds violations in data')
    parser.add_argument('--config', type=str, default='data/dataset_config.json')
    parser.add_argument('--process', type=str, default='QCD')
    parser.add_argument('--n-samples', type=int, default=50000)
    parser.add_argument('--lower', type=float, default=-4.0)
    parser.add_argument('--upper', type=float, default=4.0)
    parser.add_argument('--batch-size', type=int, default=2048)
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        dataset_config = json.load(f)
    
    # Load dataset
    print(f"Loading {args.process} dataset...")
    dataset = JetDataset(dataset_config[args.process]["path"])
    
    # Sample data
    from torch.utils.data import RandomSampler
    sampler = RandomSampler(dataset, replacement=False, num_samples=min(args.n_samples, len(dataset)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    # Collect data
    print(f"Collecting {args.n_samples} samples...")
    data_list = []
    for batch in tqdm(dataloader):
        features = batch[0] if isinstance(batch, (tuple, list)) else batch
        data_list.append(features.cpu().numpy())
    
    data = np.concatenate(data_list, axis=0)
    print(f"Data shape: {data.shape}")
    
    # Check bounds
    print(f"\nChecking bounds [{args.lower}, {args.upper}]...")
    results = check_bounds(data, args.lower, args.upper)
    
    # Print results
    print("\n" + "="*80)
    print("BOUNDS CHECK RESULTS")
    print("="*80)
    print(f"Dataset: {args.process}")
    print(f"Samples: {results['n_samples']}")
    print(f"Dimensions: {results['n_dims']}")
    print(f"Bounds: {results['bounds']}")
    print(f"\nData range: [{results['min_value']:.4f}, {results['max_value']:.4f}]")
    print(f"\nSamples with ANY feature out of bounds: {results['n_samples_with_violations']} ({results['pct_samples_with_violations']:.2f}%)")
    print(f"Total feature violations: {results['n_total_violations']} ({results['pct_features_violated']:.4f}%)")
    print(f"Average violations per violated sample: {results['mean_violations_per_violated_sample']:.2f}")
    print(f"\nTop 5 dimensions with most violations:")
    for dim, count in results['dims_with_most_violations']:
        pct = 100 * count / results['n_samples']
        print(f"  Dimension {dim}: {count} violations ({pct:.2f}%)")
    print("="*80)
    
    # Key insight
    if results['pct_samples_with_violations'] > 50:
        print(f"\n⚠️  WARNING: {results['pct_samples_with_violations']:.1f}% of samples have out-of-bounds features!")
        print("   This means MCMC with hard bounds will frequently clip/reject samples.")
        print("   Consider:")
        print("   - Using wider bounds (e.g., [-5, 5] or [-6, 6])")
        print("   - Removing bounds entirely")
        print("   - Using soft penalties instead of hard clipping")


if __name__ == '__main__':
    main()
