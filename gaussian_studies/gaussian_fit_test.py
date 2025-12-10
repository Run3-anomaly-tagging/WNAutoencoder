"""
Test how well the input data fits a Gaussian distribution.

Fits the data to a multivariate Gaussian using empirical mean and covariance,
then performs multiple statistical tests to assess goodness of fit:
- Kolmogorov-Smirnov test for each dimension
- Anderson-Darling test for normality
- Chi-squared test for multivariate Gaussian (via Mahalanobis distances)
- Q-Q plots for visual inspection
- Mahalanobis distance distribution
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import mahalanobis
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils.jet_dataset import JetDataset


def compute_statistics(data):
    """Compute mean and covariance of data."""
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    return mean, cov


def ks_test_per_dimension(data, mean=None, std=None):
    """
    Perform Kolmogorov-Smirnov test for normal distribution
    for each dimension independently.
    
    Args:
        data: Data array (n_samples, n_dims)
        mean: Mean vector for each dimension (uses empirical mean if None)
        std: Std vector for each dimension (uses empirical std if None)
    
    Returns:
        dict: Dictionary with dimension indices as keys and tuples of (statistic, p-value)
    """
    n_dims = data.shape[1]
    results = {}
    
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    
    for dim in range(n_dims):
        dim_data = data[:, dim]
        # Test against normal with empirical mean and std
        statistic, p_value = stats.kstest(dim_data, 'norm', args=(mean[dim], std[dim]))
        results[dim] = (statistic, p_value)
    
    return results


def anderson_test_per_dimension(data, standardize=True):
    """
    Perform Anderson-Darling test for normality for each dimension.
    
    Args:
        data: Data array (n_samples, n_dims)
        standardize: If True, standardize each dimension before testing (required for AD test)
    
    Returns:
        dict: Dictionary with dimension indices as keys and test results
    """
    n_dims = data.shape[1]
    results = {}
    
    for dim in range(n_dims):
        dim_data = data[:, dim]
        if standardize:
            # Standardize to mean=0, std=1 for testing (AD test requires this)
            dim_data = (dim_data - np.mean(dim_data)) / np.std(dim_data)
        result = stats.anderson(dim_data, dist='norm')
        results[dim] = result
    
    return results


def compute_mahalanobis_distances(data, mean, cov):
    """
    Compute Mahalanobis distances from the fitted Gaussian center.
    Should follow a chi-squared distribution with k degrees of freedom if data is Gaussian.
    
    Args:
        data: Data points (n_samples, n_dims)
        mean: Fitted mean vector
        cov: Fitted covariance matrix
    
    Returns:
        array: Squared Mahalanobis distances
    """
    cov_inv = np.linalg.inv(cov)
    
    # Compute squared Mahalanobis distances
    diff = data - mean
    distances_sq = np.sum(diff @ cov_inv * diff, axis=1)
    
    return distances_sq


def chi_squared_test_mahalanobis(mahal_distances_sq, n_dims, n_bins=50):
    """
    """
    # Chi-squared test
    # Expected distribution is chi-squared with n_dims degrees of freedom
    max_dist = np.percentile(mahal_distances_sq, 99)  # Use 99th percentile to avoid outliers
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    
    observed_freq, _ = np.histogram(mahal_distances_sq, bins=bin_edges)
    
    # Expected frequencies
    expected_freq = np.zeros(n_bins)
    for i in range(n_bins):
        # Probability mass in each bin
        p_low = stats.chi2.cdf(bin_edges[i], df=n_dims)
        p_high = stats.chi2.cdf(bin_edges[i+1], df=n_dims)
        expected_freq[i] = (p_high - p_low) * len(mahal_distances_sq)
    
    # Remove bins with very low expected counts
    mask = expected_freq >= 5
    observed_freq_filtered = observed_freq[mask]
    expected_freq_filtered = expected_freq[mask]
    
    # Normalize to ensure they sum to the same value
    observed_sum = np.sum(observed_freq_filtered)
    expected_freq_filtered = expected_freq_filtered * (observed_sum / np.sum(expected_freq_filtered))
    
    # Chi-squared test
    chi2_statistic, p_value = stats.chisquare(observed_freq_filtered, expected_freq_filtered)
    
    return chi2_statistic, p_value, observed_freq, expected_freq, bin_edges

def plot_qq_plots(data, mean, std, output_dir, n_plot_dims=16):
    """
    Create Q-Q plots for visual inspection of normality.
    Plots the first n_plot_dims dimensions against fitted normal.
    """
    n_dims = min(n_plot_dims, data.shape[1])
    n_rows = int(np.ceil(np.sqrt(n_dims)))
    n_cols = int(np.ceil(n_dims / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_dims > 1 else [axes]
    
    for dim in range(n_dims):
        ax = axes[dim]
        # Standardize using fitted parameters
        standardized = (data[:, dim] - mean[dim]) / std[dim]
        stats.probplot(standardized, dist="norm", plot=ax)
        ax.set_title(f'Dimension {dim}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for dim in range(n_dims, len(axes)):
        axes[dim].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qq_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Q-Q plots saved to {output_dir}/qq_plots.png")


def plot_marginal_distributions(data, mean, std, output_dir, n_plot_dims=16):
    """
    Plot marginal distributions for each dimension with overlaid fitted normal.
    """
    n_dims = min(n_plot_dims, data.shape[1])
    n_rows = int(np.ceil(np.sqrt(n_dims)))
    n_cols = int(np.ceil(n_dims / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_dims > 1 else [axes]
    
    for dim in range(n_dims):
        ax = axes[dim]
        dim_data = data[:, dim]
        
        # Plot histogram
        ax.hist(dim_data, bins=50, density=True, alpha=0.6, label='Data')
        
        # Plot fitted normal
        x = np.linspace(dim_data.min(), dim_data.max(), 1000)
        fitted_normal = stats.norm.pdf(x, mean[dim], std[dim])
        ax.plot(x, fitted_normal, 'r-', lw=2, label=f'N({mean[dim]:.2f},{std[dim]:.2f})')
        
        ax.set_title(f'Dimension {dim}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for dim in range(n_dims, len(axes)):
        axes[dim].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'marginal_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Marginal distributions saved to {output_dir}/marginal_distributions.png")


def plot_mahalanobis_distribution(mahal_distances_sq, n_dims, output_dir):
    """
    Plot the distribution of squared Mahalanobis distances vs expected chi-squared distribution.
    """
    # Histogram with overlay
    fig, ax = plt.subplots(figsize=(8, 6))
    
    max_dist = np.percentile(mahal_distances_sq, 99)
    bins = np.linspace(0, max_dist, 100)
    ax.hist(mahal_distances_sq, bins=bins, density=True, alpha=0.6, label='Data')
    
    x = np.linspace(0, max_dist, 1000)
    expected = stats.chi2.pdf(x, df=n_dims)
    ax.plot(x, expected, 'r-', lw=2, label=f'χ²({n_dims})')
    ax.set_xlabel('Squared Mahalanobis Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mahalanobis_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Mahalanobis distribution plot saved to {output_dir}/mahalanobis_distribution.png")
    
    # Q-Q plot for chi-squared (separate figure)
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(mahal_distances_sq, dist=stats.chi2, sparams=(n_dims,), plot=ax)
    ax.set_title(f'Q-Q Plot vs χ²({n_dims})', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mahalanobis_qq_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Mahalanobis Q-Q plot saved to {output_dir}/mahalanobis_qq_plot.png")


def plot_covariance_matrix(cov, output_dir):
    """
    Plot the empirical covariance matrix with hexbin for off-diagonal elements.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap of covariance matrix
    im1 = ax1.imshow(cov, cmap='RdBu_r', aspect='auto', vmin=-0.2, vmax=0.2)
    ax1.set_title('Empirical Covariance Matrix')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Dimension')
    plt.colorbar(im1, ax=ax1, label='Covariance')
    
    # Extract off-diagonal elements
    n_dims = cov.shape[0]
    indices = np.triu_indices(n_dims, k=1)
    off_diag = cov[indices]
    
    # Histogram of off-diagonal covariances
    ax2.hist(off_diag, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', lw=2, label='Zero')
    ax2.set_xlabel('Off-diagonal Covariance')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Distribution of Off-diagonal Elements\n(mean={np.mean(off_diag):.4f}, std={np.std(off_diag):.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'covariance_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Covariance matrix plot saved to {output_dir}/covariance_matrix.png")


def print_test_summary(ks_results, anderson_results, mean, cov, mahal_chi2, mahal_pval):
    """
    Print a comprehensive summary of all statistical tests.
    """
    n_dims = len(ks_results)
    
    print("\n" + "="*80)
    print("GAUSSIAN FIT TEST SUMMARY")
    print("="*80)
    
    # Basic statistics
    print("\n1. FITTED GAUSSIAN PARAMETERS")
    print("-" * 80)
    print(f"Number of dimensions: {n_dims}")
    print(f"Mean vector norm: {np.linalg.norm(mean):.6f}")
    print(f"Mean vector (first 10 dims): {mean[:10]}")
    
    diag_cov = np.diag(cov)
    print(f"Diagonal of covariance (first 10): {diag_cov[:10]}")
    print(f"Mean variance: {np.mean(diag_cov):.6f}")
    print(f"Std of variances: {np.std(diag_cov):.6f}")
    
    # Off-diagonal statistics
    indices = np.triu_indices(n_dims, k=1)
    off_diag = cov[indices]
    print(f"Mean off-diagonal covariance: {np.mean(off_diag):.6f}")
    print(f"Std off-diagonal covariance: {np.std(off_diag):.6f}")
    print(f"Max abs off-diagonal: {np.max(np.abs(off_diag)):.6f}")
    
    # KS test results
    print("\n2. KOLMOGOROV-SMIRNOV TEST (per dimension vs fitted normal)")
    print("-" * 80)
    ks_statistics = [result[0] for result in ks_results.values()]
    ks_pvalues = [result[1] for result in ks_results.values()]
    
    # Count how many pass at different significance levels
    alpha_01 = np.sum(np.array(ks_pvalues) > 0.01)
    alpha_05 = np.sum(np.array(ks_pvalues) > 0.05)
    
    print(f"Mean KS statistic: {np.mean(ks_statistics):.6f}")
    print(f"Mean p-value: {np.mean(ks_pvalues):.6f}")
    print(f"Dimensions passing at α=0.05: {alpha_05}/{n_dims} ({100*alpha_05/n_dims:.1f}%)")
    print(f"Dimensions passing at α=0.01: {alpha_01}/{n_dims} ({100*alpha_01/n_dims:.1f}%)")
    
    # Show worst dimensions
    worst_dims = np.argsort(ks_pvalues)[:5]
    print("\nWorst 5 dimensions (lowest p-values):")
    for dim in worst_dims:
        stat, pval = ks_results[dim]
        print(f"  Dim {dim}: statistic={stat:.6f}, p-value={pval:.6f}")
    
    # Anderson-Darling test
    print("\n3. ANDERSON-DARLING TEST (per dimension, standardized)")
    print("-" * 80)
    # AD critical values at different significance levels: [15%, 10%, 5%, 2.5%, 1%]
    reject_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Count rejections at each level
    
    for dim, result in anderson_results.items():
        for i, crit_val in enumerate(result.critical_values):
            if result.statistic > crit_val:
                reject_counts[i] += 1
    
    sig_levels = [15, 10, 5, 2.5, 1]
    print("Rejection rates at different significance levels:")
    for i, sig in enumerate(sig_levels):
        print(f"  α={sig}%: {reject_counts[i]}/{n_dims} rejected ({100*reject_counts[i]/n_dims:.1f}%)")
    
    # Mahalanobis distance test
    print("\n4. MULTIVARIATE TEST (Mahalanobis distance vs χ²)")
    print("-" * 80)
    print(f"Chi-squared statistic: {mahal_chi2:.6f}")
    print(f"P-value: {mahal_pval:.6f}")
    if mahal_pval > 0.05:
        print("Result: PASS (Mahalanobis distances follow χ² distribution at α=0.05)")
    else:
        print("Result: REJECT (Mahalanobis distances deviate from χ² distribution at α=0.05)")
    
    # Overall assessment
    print("\n5. OVERALL ASSESSMENT")
    print("-" * 80)
    
    criteria = {
        "KS test pass rate > 90% (α=0.05)": alpha_05 / n_dims > 0.9,
        "KS test pass rate > 95% (α=0.01)": alpha_01 / n_dims > 0.95,
        "AD test: <10% rejected (α=5%)": reject_counts[2] / n_dims < 0.1,
        "Mahalanobis test pass (α=0.05)": mahal_pval > 0.05,
        "Low correlation (max |r| < 0.3)": np.max(np.abs(off_diag / np.outer(np.sqrt(diag_cov), np.sqrt(diag_cov))[indices])) < 0.3
    }
    
    for criterion, passed in criteria.items():
        status = "✓" if passed else "✗"
        print(f"{status} {criterion}")
    
    overall_pass = sum(criteria.values()) >= 3
    print("\n" + "="*80)
    if overall_pass:
        print("CONCLUSION: Data fits a multivariate Gaussian reasonably well")
    else:
        print("CONCLUSION: Data shows significant deviations from Gaussian")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Test Gaussian fit of input data')
    parser.add_argument('--config', type=str, default='data/dataset_config.json',
                        help='Path to dataset config JSON')
    parser.add_argument('--process', type=str, default='QCD',
                        help='Process name from dataset config')
    parser.add_argument('--n-samples', type=int, default=50000,
                        help='Number of samples to use for testing')
    parser.add_argument('--output-dir', type=str, default='results/gaussian_studies/fit_test',
                        help='Output directory for plots and results')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size for data loading')
    parser.add_argument('--n-plot-dims', type=int, default=16,
                        help='Number of dimensions to plot in detail')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset config
    config_path = os.path.join(project_root, args.config)
    with open(config_path, 'r') as f:
        dataset_config = json.load(f)
    
    # Load dataset
    print(f"Loading {args.process} dataset...")
    dataset = JetDataset(dataset_config[args.process]["path"])
    
    # Sample data
    sampler = RandomSampler(dataset, replacement=False, num_samples=min(args.n_samples, len(dataset)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    # Collect data
    print(f"Collecting {args.n_samples} samples...")
    data_list = []
    for batch in tqdm(dataloader):
        # JetDataset returns (features, features) tuple
        features = batch[0] if isinstance(batch, (tuple, list)) else batch
        data_list.append(features.cpu().numpy())
    
    data = np.concatenate(data_list, axis=0)
    n_samples, n_dims = data.shape
    print(f"Data shape: {data.shape}")
    
    # Compute basic statistics (fit Gaussian)
    print("\nFitting Gaussian distribution...")
    mean, cov = compute_statistics(data)
    std = np.sqrt(np.diag(cov))
    
    print(f"Fitted mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Fitted std range: [{std.min():.4f}, {std.max():.4f}]")
    
    # Perform KS tests against fitted normal
    print("Performing Kolmogorov-Smirnov tests...")
    ks_results = ks_test_per_dimension(data, mean=mean, std=std)
    
    # Perform Anderson-Darling tests
    print("Performing Anderson-Darling tests...")
    anderson_results = anderson_test_per_dimension(data, standardize=True)
    
    # Compute Mahalanobis distances using fitted parameters
    print("Computing Mahalanobis distances...")
    mahal_distances_sq = compute_mahalanobis_distances(data, mean=mean, cov=cov)
    
    # Chi-squared test on Mahalanobis distances
    print("Performing chi-squared test on Mahalanobis distances...")
    mahal_chi2, mahal_pval, _, _, _ = chi_squared_test_mahalanobis(mahal_distances_sq, n_dims)
    
    # Print summary
    print_test_summary(ks_results, anderson_results, mean, cov, mahal_chi2, mahal_pval)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_qq_plots(data, mean, std, output_dir, args.n_plot_dims)
    plot_marginal_distributions(data, mean, std, output_dir, args.n_plot_dims)
    plot_mahalanobis_distribution(mahal_distances_sq, n_dims, output_dir)
    plot_covariance_matrix(cov, output_dir)
    
    # Save results to JSON
    results = {
        'n_samples': int(n_samples),
        'n_dims': int(n_dims),
        'fitted_mean_norm': float(np.linalg.norm(mean)),
        'fitted_mean_range': [float(mean.min()), float(mean.max())],
        'fitted_std_range': [float(std.min()), float(std.max())],
        'mean_variance': float(np.mean(np.diag(cov))),
        'std_variance': float(np.std(np.diag(cov))),
        'mean_off_diagonal': float(np.mean(cov[np.triu_indices(n_dims, k=1)])),
        'max_abs_off_diagonal': float(np.max(np.abs(cov[np.triu_indices(n_dims, k=1)]))),
        'ks_mean_statistic': float(np.mean([r[0] for r in ks_results.values()])),
        'ks_mean_pvalue': float(np.mean([r[1] for r in ks_results.values()])),
        'ks_pass_rate_alpha_05': float(np.sum([r[1] > 0.05 for r in ks_results.values()]) / n_dims),
        'ks_pass_rate_alpha_01': float(np.sum([r[1] > 0.01 for r in ks_results.values()]) / n_dims),
        'mahalanobis_chi2': float(mahal_chi2),
        'mahalanobis_pvalue': float(mahal_pval)
    }
    
    results_path = os.path.join(output_dir, 'gaussian_fit_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
