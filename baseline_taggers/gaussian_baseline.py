"""
Gaussian Baseline Tagger for Anomaly Detection

Uses L2 norm (||x||^2) as anomaly score, assuming features follow N(0, I).
Since QCD features are scaled to zero mean and unit variance, this provides
a simple baseline for comparison with WNAE models.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, SequentialSampler
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.jet_dataset import JetDataset


def compute_l2_norm_scores(dataloader):
    """
    Compute L2 norm squared (||x||^2) for each jet.
    
    Args:
        dataloader: DataLoader yielding jet feature batches
        
    Returns:
        np.ndarray: Array of L2 norm squared values
    """
    scores = []
    for batch in dataloader:
        x = batch[0].numpy()  # Shape: (batch_size, n_features)
        l2_squared = np.sum(x ** 2, axis=1)  # ||x||^2 for each sample
        scores.extend(l2_squared)
    return np.array(scores)


def plot_roc_curves(background_scores, signal_scores_dict, output_dir):
    """
    Plot ROC curves for all signal processes vs background.
    
    Args:
        background_scores: Anomaly scores for background (QCD)
        signal_scores_dict: Dict mapping signal name to anomaly scores
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    results = {}
    
    for signal_name, signal_scores in signal_scores_dict.items():
        # Combine background and signal
        y_true = np.concatenate([
            np.zeros(len(background_scores)),  # 0 = background
            np.ones(len(signal_scores))         # 1 = signal
        ])
        y_scores = np.concatenate([background_scores, signal_scores])
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, linewidth=2, label=f'{signal_name} (AUC = {roc_auc:.3f})')
        
        results[signal_name] = {
            'auc': float(roc_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    
    plt.plot([0, 1], [0, 1], color='blue', linewidth=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Background mistag rate', fontsize=14)
    plt.ylabel('Signal efficiency', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    roc_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=300)
    print(f"ROC curves saved to: {roc_path}")
    plt.close()
    
    # Save numerical results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    return results


def plot_score_distributions(background_scores, signal_scores_dict, output_dir):
    """
    Plot anomaly score distributions for background and signals.
    
    Args:
        background_scores: Anomaly scores for background (QCD)
        signal_scores_dict: Dict mapping signal name to anomaly scores
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Determine appropriate bins
    all_scores = np.concatenate([background_scores] + list(signal_scores_dict.values()))
    bins = np.linspace(np.percentile(all_scores, 0.1), 
                       np.percentile(all_scores, 99.9), 100)
    
    # Plot background
    plt.hist(background_scores, bins=bins, histtype='step', 
             linewidth=2, label='QCD (Background)', density=True, color='blue')
    
    # Plot signals
    colors = plt.cm.Set1(np.linspace(0, 1, len(signal_scores_dict)))
    for (signal_name, signal_scores), color in zip(signal_scores_dict.items(), colors):
        plt.hist(signal_scores, bins=bins, histtype='step', 
                 linewidth=2, label=signal_name, density=True, color=color)
    
    plt.xlabel('Anomaly Score (||x||²)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Anomaly Score Distributions: Gaussian Baseline', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    dist_path = os.path.join(output_dir, 'score_distributions.png')
    plt.savefig(dist_path, dpi=300)
    print(f"Score distributions saved to: {dist_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Gaussian Baseline Tagger: Anomaly detection using L2 norm of scaled features'
    )
    parser.add_argument('--config', '-c', default='data/dataset_config.json',
                        help='Path to dataset configuration JSON')
    parser.add_argument('--input-dim', type=int, default=256,
                        help='Number of input features to use')
    parser.add_argument('--max-jets', type=int, default=20000,
                        help='Maximum number of jets to load per process')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Batch size for data loading')
    parser.add_argument('--output-dir', '-o', default='baseline_taggers/results',
                        help='Directory to save results')
    parser.add_argument('--background', default='QCD',
                        help='Background process name (default: QCD)')
    parser.add_argument('--pca', default=None,
                        help='Optional path to PCA components .npy file')
    
    args = parser.parse_args()
    
    # Load dataset configuration
    print(f"Loading dataset config from: {args.config}")
    with open(args.config, 'r') as f:
        dataset_config = json.load(f)
    
    # Automatically detect signals: all processes except the background
    all_processes = list(dataset_config.keys())
    if args.background not in all_processes:
        raise ValueError(f"Background process '{args.background}' not found in config")
    
    signals = [p for p in all_processes if p != args.background]
    
    if not signals:
        raise ValueError(f"No signal processes found in config (only {args.background} present)")
    
    print(f"\n{'='*60}")
    print(f"Gaussian Baseline Tagger Configuration")
    print(f"{'='*60}")
    print(f"Input dimensions: {args.input_dim}")
    print(f"Max jets per process: {args.max_jets}")
    print(f"Batch size: {args.batch_size}")
    print(f"PCA components: {args.pca if args.pca else 'None'}")
    print(f"Background: {args.background}")
    print(f"Signal processes: {', '.join(signals)}")
    print(f"{'='*60}\n")
    
    # Load and score background
    print(f"Loading {args.background} background...")
    qcd_path = dataset_config[args.background]['path']
    qcd_dataset = JetDataset(qcd_path, input_dim=args.input_dim, pca_components=args.pca)
    
    # Limit number of jets if requested
    if args.max_jets and len(qcd_dataset) > args.max_jets:
        indices = np.random.choice(len(qcd_dataset), args.max_jets, replace=False)
        qcd_dataset = JetDataset(qcd_path, indices=indices, 
                                  input_dim=args.input_dim, pca_components=args.pca)
    
    qcd_loader = DataLoader(qcd_dataset, batch_size=args.batch_size, 
                            sampler=SequentialSampler(qcd_dataset))
    
    background_scores = compute_l2_norm_scores(qcd_loader)
    print(f"  Loaded {len(background_scores)} {args.background} jets")
    print(f"  Score stats - Mean: {np.mean(background_scores):.2f}, "
          f"Std: {np.std(background_scores):.2f}, "
          f"Median: {np.median(background_scores):.2f}")
    
    # Load and score signals
    signal_scores_dict = {}
    for signal_name in signals:
        print(f"\nLoading {signal_name} signal...")
        signal_path = dataset_config[signal_name]['path']
        signal_dataset = JetDataset(signal_path, input_dim=args.input_dim, 
                                     pca_components=args.pca)
        
        # Limit number of jets if requested
        if args.max_jets and len(signal_dataset) > args.max_jets:
            indices = np.random.choice(len(signal_dataset), args.max_jets, replace=False)
            signal_dataset = JetDataset(signal_path, indices=indices,
                                        input_dim=args.input_dim, pca_components=args.pca)
        
        signal_loader = DataLoader(signal_dataset, batch_size=args.batch_size,
                                   sampler=SequentialSampler(signal_dataset))
        
        signal_scores = compute_l2_norm_scores(signal_loader)
        signal_scores_dict[signal_name] = signal_scores
        
        print(f"  Loaded {len(signal_scores)} {signal_name} jets")
        print(f"  Score stats - Mean: {np.mean(signal_scores):.2f}, "
              f"Std: {np.std(signal_scores):.2f}, "
              f"Median: {np.median(signal_scores):.2f}")
    
    # Generate plots and results
    print(f"\n{'='*60}")
    print("Generating ROC curves and results...")
    print(f"{'='*60}\n")
    
    results = plot_roc_curves(background_scores, signal_scores_dict, args.output_dir)
    plot_score_distributions(background_scores, signal_scores_dict, args.output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary of AUC Scores")
    print(f"{'='*60}")
    for signal_name, metrics in results.items():
        print(f"  {signal_name:20s}: AUC = {metrics['auc']:.4f}")
    print(f"{'='*60}\n")
    
    print(f"All results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
