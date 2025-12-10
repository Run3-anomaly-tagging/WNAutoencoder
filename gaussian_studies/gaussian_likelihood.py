"""
Multivariate Gaussian likelihood-based anomaly detection.

Fits a multivariate Gaussian to QCD background data and uses negative log-likelihood
as the anomaly score. Compares different covariance structures:
- Full covariance (captures all linear correlations)
- Diagonal covariance (assumes independence)
- Identity covariance (standard normal)

This serves as a baseline to compare against the WNAE energy-based model.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc
import argparse
from tqdm import tqdm
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils.jet_dataset import JetDataset


class GaussianLikelihoodTagger:
    """
    Anomaly detector based on multivariate Gaussian likelihood.
    """
    def __init__(self, covariance_type='full'):
        """
        Args:
            covariance_type: 'full', 'diagonal', or 'identity'
        """
        self.covariance_type = covariance_type
        self.mean = None
        self.cov = None
        self.cov_inv = None
        self.log_det_cov = None
        self.n_dims = None
        
    def fit(self, data):
        """
        Fit Gaussian parameters from training data.
        
        Args:
            data: Training data (n_samples, n_dims)
        """
        self.n_dims = data.shape[1]
        self.mean = np.mean(data, axis=0)
        
        if self.covariance_type == 'full':
            # Full covariance matrix
            self.cov = np.cov(data.T)
            # Add small regularization for numerical stability
            self.cov += np.eye(self.n_dims) * 1e-6
            self.cov_inv = np.linalg.inv(self.cov)
            self.log_det_cov = np.linalg.slogdet(self.cov)[1]
            
        elif self.covariance_type == 'diagonal':
            # Diagonal covariance (independent dimensions)
            variances = np.var(data, axis=0)
            self.cov = np.diag(variances)
            self.cov_inv = np.diag(1.0 / (variances + 1e-10))
            self.log_det_cov = np.sum(np.log(variances + 1e-10))
            
        elif self.covariance_type == 'identity':
            # Standard normal N(0, I)
            self.mean = np.zeros(self.n_dims)
            self.cov = np.eye(self.n_dims)
            self.cov_inv = np.eye(self.n_dims)
            self.log_det_cov = 0.0
            
        else:
            raise ValueError(f"Unknown covariance_type: {self.covariance_type}")
    
    def negative_log_likelihood(self, data):
        """
        Compute negative log-likelihood (anomaly score).
        
        NLL = 0.5 * [(x-μ)ᵀ Σ⁻¹ (x-μ) + log|Σ| + d*log(2π)]
        
        Args:
            data: Test data (n_samples, n_dims)
            
        Returns:
            array: Negative log-likelihood for each sample
        """
        diff = data - self.mean
        
        # Mahalanobis distance term
        mahal_dist_sq = np.sum(diff @ self.cov_inv * diff, axis=1)
        
        # Full negative log-likelihood
        nll = 0.5 * (mahal_dist_sq + self.log_det_cov + self.n_dims * np.log(2 * np.pi))
        
        return nll
    
    def score(self, data):
        """Alias for negative_log_likelihood."""
        return self.negative_log_likelihood(data)
    
    def save(self, filepath):
        """Save fitted model parameters."""
        model_dict = {
            'covariance_type': self.covariance_type,
            'mean': self.mean,
            'cov': self.cov,
            'cov_inv': self.cov_inv,
            'log_det_cov': self.log_det_cov,
            'n_dims': self.n_dims
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load fitted model parameters."""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        self.covariance_type = model_dict['covariance_type']
        self.mean = model_dict['mean']
        self.cov = model_dict['cov']
        self.cov_inv = model_dict['cov_inv']
        self.log_det_cov = model_dict['log_det_cov']
        self.n_dims = model_dict['n_dims']
        print(f"Model loaded from {filepath}")


def load_dataset_batched(file_path, n_samples=None, batch_size=1024):
    """Load dataset in batches."""
    from torch.utils.data import DataLoader, RandomSampler
    
    dataset = JetDataset(file_path)
    
    if n_samples is not None:
        sampler = RandomSampler(dataset, replacement=False, num_samples=min(n_samples, len(dataset)))
    else:
        sampler = None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    data_list = []
    for batch in tqdm(dataloader, desc="Loading data"):
        features = batch[0] if isinstance(batch, (tuple, list)) else batch
        data_list.append(features.cpu().numpy())
    
    return np.concatenate(data_list, axis=0)


def evaluate_tagger(tagger, bkg_data, signal_data_dict, output_dir):
    """
    Evaluate tagger performance on background and signal samples.
    
    Args:
        tagger: Fitted GaussianLikelihoodTagger
        bkg_data: Background data for evaluation
        signal_data_dict: Dict of {signal_name: signal_data}
        output_dir: Directory to save results
    """
    # Compute scores
    print(f"\nEvaluating {tagger.covariance_type} covariance tagger...")
    bkg_scores = tagger.score(bkg_data)
    
    results = {
        'covariance_type': tagger.covariance_type,
        'bkg_nll_mean': float(np.mean(bkg_scores)),
        'bkg_nll_std': float(np.std(bkg_scores)),
        'signals': {}
    }
    
    # Plot score distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (sig_name, sig_data) in enumerate(signal_data_dict.items()):
        if idx >= 6:
            break
        
        sig_scores = tagger.score(sig_data)
        
        # Compute ROC curve
        y_true = np.concatenate([np.zeros(len(bkg_scores)), np.ones(len(sig_scores))])
        y_scores = np.concatenate([bkg_scores, sig_scores])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        results['signals'][sig_name] = {
            'auc': float(roc_auc),
            'nll_mean': float(np.mean(sig_scores)),
            'nll_std': float(np.std(sig_scores))
        }
        
        # Plot score distribution
        ax = axes[idx]
        bins = np.linspace(min(bkg_scores.min(), sig_scores.min()), 
                          max(bkg_scores.max(), sig_scores.max()), 50)
        ax.hist(bkg_scores, bins=bins, alpha=0.5, label='QCD', density=True)
        ax.hist(sig_scores, bins=bins, alpha=0.5, label=sig_name, density=True)
        ax.set_xlabel('Negative Log-Likelihood')
        ax.set_ylabel('Density')
        ax.set_title(f'{sig_name} (AUC={roc_auc:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"  {sig_name}: AUC = {roc_auc:.4f}")
    
    # Hide unused subplots
    for idx in range(len(signal_data_dict), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'score_distributions_{tagger.covariance_type}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Score distributions saved to {plot_path}")
    
    # Plot ROC curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for sig_name, sig_data in signal_data_dict.items():
        sig_scores = tagger.score(sig_data)
        
        y_true = np.concatenate([np.zeros(len(bkg_scores)), np.ones(len(sig_scores))])
        y_scores = np.concatenate([bkg_scores, sig_scores])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, label=f'{sig_name} (AUC={roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves ({tagger.covariance_type} covariance)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    roc_path = os.path.join(output_dir, f'roc_curves_{tagger.covariance_type}.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {roc_path}")
    
    return results


def compare_covariance_types(results_list, output_dir):
    """
    Compare performance across different covariance types.
    """
    # Extract AUCs for each signal across covariance types
    signal_names = list(results_list[0]['signals'].keys())
    cov_types = [r['covariance_type'] for r in results_list]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(signal_names))
    width = 0.25
    
    for idx, results in enumerate(results_list):
        aucs = [results['signals'][sig]['auc'] for sig in signal_names]
        ax.bar(x + idx * width, aucs, width, label=results['covariance_type'])
    
    ax.set_xlabel('Signal Process')
    ax.set_ylabel('AUC')
    ax.set_title('Comparison of Covariance Types')
    ax.set_xticks(x + width)
    ax.set_xticklabels(signal_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    comp_path = os.path.join(output_dir, 'covariance_comparison.png')
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {comp_path}")


def main():
    parser = argparse.ArgumentParser(description='Gaussian likelihood baseline tagger')
    parser.add_argument('--config', type=str, default='data/dataset_config.json',
                        help='Path to dataset config JSON')
    parser.add_argument('--bkg-process', type=str, default='QCD',
                        help='Background process name')
    parser.add_argument('--signal-processes', type=str, nargs='+',
                        default=['GluGluHto2B', 'SVJ', 'Yto4Q', 'TTto4Q'],
                        help='List of signal process names')
    parser.add_argument('--n-train', type=int, default=100000,
                        help='Number of background samples for training')
    parser.add_argument('--n-test', type=int, default=50000,
                        help='Number of samples for testing')
    parser.add_argument('--output-dir', type=str, default='results/gaussian_studies/likelihood',
                        help='Output directory')
    parser.add_argument('--covariance-types', type=str, nargs='+',
                        default=['full', 'diagonal', 'identity'],
                        help='Covariance types to test')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch size for data loading')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset config
    config_path = os.path.join(project_root, args.config)
    with open(config_path, 'r') as f:
        dataset_config = json.load(f)
    
    # Load training data (background)
    print(f"Loading {args.n_train} training samples from {args.bkg_process}...")
    train_data = load_dataset_batched(
        dataset_config[args.bkg_process]["path"],
        n_samples=args.n_train,
        batch_size=args.batch_size
    )
    print(f"Training data shape: {train_data.shape}")
    
    # Load test data (background)
    print(f"\nLoading {args.n_test} test samples from {args.bkg_process}...")
    test_bkg_data = load_dataset_batched(
        dataset_config[args.bkg_process]["path"],
        n_samples=args.n_test,
        batch_size=args.batch_size
    )
    
    # Load signal data
    signal_data_dict = {}
    for sig_process in args.signal_processes:
        if sig_process not in dataset_config:
            print(f"Warning: {sig_process} not in config, skipping")
            continue
        
        print(f"Loading {args.n_test} samples from {sig_process}...")
        sig_data = load_dataset_batched(
            dataset_config[sig_process]["path"],
            n_samples=args.n_test,
            batch_size=args.batch_size
        )
        signal_data_dict[sig_process] = sig_data
    
    # Train and evaluate taggers with different covariance types
    all_results = []
    
    for cov_type in args.covariance_types:
        print(f"\n{'='*80}")
        print(f"Training {cov_type} covariance tagger")
        print(f"{'='*80}")
        
        # Fit tagger
        tagger = GaussianLikelihoodTagger(covariance_type=cov_type)
        tagger.fit(train_data)
        
        # Save model
        model_path = os.path.join(output_dir, f'gaussian_tagger_{cov_type}.pkl')
        tagger.save(model_path)
        
        # Evaluate
        results = evaluate_tagger(tagger, test_bkg_data, signal_data_dict, output_dir)
        all_results.append(results)
    
    # Save all results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll results saved to {results_path}")
    
    # Compare covariance types
    if len(all_results) > 1:
        compare_covariance_types(all_results, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for results in all_results:
        print(f"\n{results['covariance_type'].upper()} Covariance:")
        print(f"  Background NLL: {results['bkg_nll_mean']:.4f} ± {results['bkg_nll_std']:.4f}")
        for sig_name, sig_results in results['signals'].items():
            print(f"  {sig_name}: AUC = {sig_results['auc']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
