"""
Energy Tail Analysis for deep_bottleneck_qcd_wnae_CFG1

Investigates which QCD jets end up in high-energy tail of E+ distribution,
while MCMC samples concentrate in low-energy core.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.jet_dataset import JetDataset
from wnae import WNAE
from model_config.model_registry import MODEL_REGISTRY
from model_config.model_config import WNAE_PARAM_PRESETS

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "deep_bottleneck_qcd"
WNAE_PRESET = "CFG1"
CHECKPOINT_PATH = "../models/deep_bottleneck_qcd_wnae_CFG1/checkpoint.pth"
DATASET_CONFIG = "../data/dataset_config.json"
OUTPUT_DIR = "./plots"
DATA_DIR = "./data"

BATCH_SIZE = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Percentile thresholds for analysis
TAIL_PERCENTILES = [80, 90]
CORE_PERCENTILE = 50

# ============================================================================
# Helper Functions
# ============================================================================

def load_model_and_data(checkpoint_path, model_name, dataset_config, preset):
    """Load trained WNAE model and QCD validation dataset."""
    
    print(f"Loading model: {model_name} with preset {preset}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Get model configuration
    model_config = MODEL_REGISTRY[model_name]
    input_dim = model_config["input_dim"]
    encoder = model_config["encoder"]()
    decoder = model_config["decoder"]()
    
    # Get WNAE parameters
    wnae_params = WNAE_PARAM_PRESETS[preset]
    
    # Initialize model
    model = WNAE(
        encoder=encoder,
        decoder=decoder,
        **wnae_params
    ).to(DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load replay buffer from checkpoint
    if "buffer" in checkpoint:
        print("[INFO] Loading replay buffer from checkpoint")
        stored_buffer = checkpoint["buffer"]
        try:
            if model.buffer.max_samples != len(stored_buffer):
                print(f"Stored buffer len ({len(stored_buffer)}) different from declared buffer size {model.buffer.max_samples}. Truncating/using prefix.")
                model.buffer.buffer = stored_buffer[:model.buffer.max_samples]
            else:
                model.buffer.buffer = stored_buffer
        except Exception as e:
            print(f"Failed to set model.buffer from checkpoint: {e}")
    else:
        print("No replay buffer found in checkpoint.")
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Replay buffer size: {len(model.buffer)}")
    
    # Load dataset
    with open(dataset_config, 'r') as f:
        data_config = json.load(f)
    
    process_name = model_config["process"]
    data_path = data_config[process_name]["path"]
    
    # Get PCA components if specified
    pca_components = None
    if "pca" in model_config and model_config["pca"] is not None:
        pca_path = model_config["pca"]
        pca_components = np.load(pca_path)
        print(f"Loaded PCA components: {pca_components.shape}")
    
    # Load full dataset and split
    full_dataset = JetDataset(data_path, pca_components=pca_components)
    
    # Use same split logic as training (80/20)
    np.random.seed(0)
    n_samples = len(full_dataset)
    indices = np.random.permutation(n_samples)
    train_size = int(0.8 * n_samples)
    val_indices = indices[train_size:]
    
    val_dataset = Subset(full_dataset, val_indices)
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return model, val_dataset, wnae_params


def compute_energies_batched(model, dataset, batch_size=2048):
    """Compute energies for all samples in dataset."""
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_energies = []
    all_samples = []
    
    print("Computing energies for validation set...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Handle both tuple/list and tensor returns from dataloader
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(DEVICE)
            # Energy is reconstruction error in autoencoder
            # __energy_with_samples returns (energy, latent_z, reconstructed_samples)
            energy, _, _ = model._WNAE__energy_with_samples(batch)
            all_energies.append(energy.cpu())
            all_samples.append(batch.cpu())
    
    energies = torch.cat(all_energies).numpy()
    samples = torch.cat(all_samples).numpy()
    
    return energies, samples


def get_negative_samples_and_energies(model):
    """Extract negative samples from replay buffer and compute their energies."""
    
    print("Extracting negative samples from replay buffer...")
    if len(model.buffer) == 0:
        print("Warning: Replay buffer is empty!")
        return None, None
    
    # Get samples from buffer - buffer.buffer is a list of tensors
    buffer_samples = torch.stack(model.buffer.buffer).to(DEVICE)
    
    with torch.no_grad():
        energies, _, _ = model._WNAE__energy_with_samples(buffer_samples)
    
    return energies.cpu().numpy(), buffer_samples.cpu().numpy()


def identify_tail_and_core_jets(energies, samples, tail_percentiles, core_percentile):
    """Identify jets in high-energy tail vs low-energy core."""
    
    results = {}
    
    # Core jets (low energy)
    core_threshold = np.percentile(energies, core_percentile)
    core_mask = energies < core_threshold
    results['core'] = {
        'threshold': core_threshold,
        'mask': core_mask,
        'indices': np.where(core_mask)[0],
        'samples': samples[core_mask],
        'energies': energies[core_mask]
    }
    
    # Tail jets (high energy)
    for percentile in tail_percentiles:
        tail_threshold = np.percentile(energies, percentile)
        tail_mask = energies > tail_threshold
        results[f'tail_{percentile}'] = {
            'threshold': tail_threshold,
            'mask': tail_mask,
            'indices': np.where(tail_mask)[0],
            'samples': samples[tail_mask],
            'energies': energies[tail_mask]
        }
    
    print(f"\nEnergy statistics:")
    print(f"  Mean: {energies.mean():.4f}")
    print(f"  Std: {energies.std():.4f}")
    print(f"  Min: {energies.min():.4f}")
    print(f"  Max: {energies.max():.4f}")
    print(f"\nCore (<{core_percentile}th percentile): {core_mask.sum()} jets, E < {core_threshold:.4f}")
    for percentile in tail_percentiles:
        tail_mask = results[f'tail_{percentile}']['mask']
        tail_threshold = results[f'tail_{percentile}']['threshold']
        print(f"Tail (>{percentile}th percentile): {tail_mask.sum()} jets, E > {tail_threshold:.4f}")
    
    return results


def analyze_feature_differences(core_samples, tail_samples, feature_dim):
    """Analyze which features differ most between core and tail."""
    
    print("\nAnalyzing feature differences...")
    
    feature_stats = []
    for i in range(feature_dim):
        core_feat = core_samples[:, i]
        tail_feat = tail_samples[:, i]
        
        # Compute statistics
        core_mean, core_std = core_feat.mean(), core_feat.std()
        tail_mean, tail_std = tail_feat.mean(), tail_feat.std()
        
        # KS test for distribution difference
        ks_stat, ks_pval = stats.ks_2samp(core_feat, tail_feat)
        
        # T-test for mean difference
        t_stat, t_pval = stats.ttest_ind(core_feat, tail_feat)
        
        feature_stats.append({
            'feature_idx': i,
            'core_mean': core_mean,
            'core_std': core_std,
            'tail_mean': tail_mean,
            'tail_std': tail_std,
            'mean_diff': abs(tail_mean - core_mean),
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            't_stat': t_stat,
            't_pval': t_pval
        })
    
    # Sort by KS statistic (largest difference)
    feature_stats.sort(key=lambda x: x['ks_stat'], reverse=True)
    
    return feature_stats


def compute_reconstruction_errors(model, samples):
    """Compute per-sample and per-feature reconstruction errors."""
    
    samples_tensor = torch.tensor(samples, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        reconstructed = model.reconstruct(samples_tensor)
        mse_per_sample = ((samples_tensor - reconstructed) ** 2).mean(dim=1)
        mse_per_feature = ((samples_tensor - reconstructed) ** 2).mean(dim=0)
    
    return mse_per_sample.cpu().numpy(), mse_per_feature.cpu().numpy()


def compute_latent_codes(model, samples):
    """Extract latent space representations."""
    
    samples_tensor = torch.tensor(samples, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        latent = model.encoder(samples_tensor)
    
    return latent.cpu().numpy()


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_energy_distributions(pos_energies, neg_energies, tail_thresholds, output_path):
    """Plot positive and negative energy distributions with tail markers."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine common bins
    all_energies = np.concatenate([pos_energies, neg_energies])
    bins = np.linspace(all_energies.min(), np.percentile(all_energies, 99), 100)
    
    # Plot histograms
    ax.hist(pos_energies, bins=bins, alpha=0.6, label='Positive (Real QCD)', 
            density=True, color='blue', edgecolor='black', linewidth=0.5)
    ax.hist(neg_energies, bins=bins, alpha=0.6, label='Negative (MCMC)', 
            density=True, color='red', edgecolor='black', linewidth=0.5)
    
    # Mark tail thresholds
    colors = ['orange', 'darkred']
    for i, (name, threshold) in enumerate(tail_thresholds.items()):
        ax.axvline(threshold, color=colors[i], linestyle='--', linewidth=2,
                  label=f'{name} (E={threshold:.3f})')
    
    ax.set_xlabel('Energy', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Energy Distributions: Positive vs Negative Samples', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_distributions(core_samples, tail_samples, top_features, output_path):
    """Plot distributions of top differentiating features."""
    
    n_features = min(8, len(top_features))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, feat_info in enumerate(top_features[:n_features]):
        feat_idx = feat_info['feature_idx']
        ax = axes[i]
        
        core_vals = core_samples[:, feat_idx]
        tail_vals = tail_samples[:, feat_idx]
        
        bins = np.linspace(min(core_vals.min(), tail_vals.min()),
                          max(core_vals.max(), tail_vals.max()), 50)
        
        ax.hist(core_vals, bins=bins, alpha=0.6, label='Core', density=True, color='green')
        ax.hist(tail_vals, bins=bins, alpha=0.6, label='Tail', density=True, color='red')
        
        ax.set_xlabel(f'Feature {feat_idx}', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Feature {feat_idx} (KS={feat_info["ks_stat"]:.3f})', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Feature Distributions: Core vs Tail Jets', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_importance(feature_stats, output_path):
    """Plot feature importance ranked by KS statistic."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top 20 features by KS stat
    top_features = feature_stats[:20]
    feat_indices = [f['feature_idx'] for f in top_features]
    ks_stats = [f['ks_stat'] for f in top_features]
    mean_diffs = [f['mean_diff'] for f in top_features]
    
    axes[0].barh(range(len(feat_indices)), ks_stats, color='steelblue')
    axes[0].set_yticks(range(len(feat_indices)))
    axes[0].set_yticklabels(feat_indices)
    axes[0].set_xlabel('KS Statistic', fontsize=12)
    axes[0].set_ylabel('Feature Index', fontsize=12)
    axes[0].set_title('Top 20 Features by Distribution Difference', fontsize=13)
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    axes[1].barh(range(len(feat_indices)), mean_diffs, color='coral')
    axes[1].set_yticks(range(len(feat_indices)))
    axes[1].set_yticklabels(feat_indices)
    axes[1].set_xlabel('|Mean Difference|', fontsize=12)
    axes[1].set_ylabel('Feature Index', fontsize=12)
    axes[1].set_title('Top 20 Features by Mean Difference', fontsize=13)
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_reconstruction_comparison(core_mse_feat, tail_mse_feat, output_path):
    """Compare reconstruction errors per feature for core vs tail."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Per-feature MSE
    axes[0].plot(core_mse_feat, label='Core', alpha=0.7, linewidth=1.5)
    axes[0].plot(tail_mse_feat, label='Tail', alpha=0.7, linewidth=1.5)
    axes[0].set_xlabel('Feature Index', fontsize=12)
    axes[0].set_ylabel('Mean Squared Error', fontsize=12)
    axes[0].set_title('Reconstruction Error per Feature', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Difference
    diff = tail_mse_feat - core_mse_feat
    axes[1].bar(range(len(diff)), diff, color=['red' if d > 0 else 'green' for d in diff], alpha=0.7)
    axes[1].set_xlabel('Feature Index', fontsize=12)
    axes[1].set_ylabel('MSE Difference (Tail - Core)', fontsize=12)
    axes[1].set_title('Reconstruction Error Difference', fontsize=13)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_latent_space(core_latent, tail_latent, output_path, method='pca'):
    """Visualize latent space separation between core and tail."""
    
    # Combine for consistent projection
    all_latent = np.vstack([core_latent, tail_latent])
    labels = np.array([0] * len(core_latent) + [1] * len(tail_latent))
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(all_latent)
        title_suffix = f'(PCA: {reducer.explained_variance_ratio_[:2].sum()*100:.1f}% var)'
    else:  # t-SNE
        # Subsample for speed
        max_samples = 5000
        if len(all_latent) > max_samples:
            indices = np.random.choice(len(all_latent), max_samples, replace=False)
            all_latent = all_latent[indices]
            labels = labels[indices]
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(all_latent)
        title_suffix = '(t-SNE)'
    
    # Split back
    core_reduced = reduced[labels == 0]
    tail_reduced = reduced[labels == 1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(core_reduced[:, 0], core_reduced[:, 1], 
              alpha=0.4, s=10, label='Core', color='green')
    ax.scatter(tail_reduced[:, 0], tail_reduced[:, 1], 
              alpha=0.4, s=10, label='Tail', color='red')
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(f'Latent Space: Core vs Tail Jets {title_suffix}', fontsize=14)
    ax.legend(fontsize=11, markerscale=3)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_energy_correlations(energies, samples, mse_per_sample, output_path):
    """Plot correlations between energy and other quantities."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Energy vs MSE
    axes[0].scatter(energies, mse_per_sample, alpha=0.3, s=5)
    axes[0].set_xlabel('Energy', fontsize=12)
    axes[0].set_ylabel('Total MSE', fontsize=12)
    axes[0].set_title('Energy vs Reconstruction Error', fontsize=13)
    axes[0].grid(alpha=0.3)
    
    # Correlation coefficient
    corr = np.corrcoef(energies, mse_per_sample)[0, 1]
    axes[0].text(0.05, 0.95, f'Corr: {corr:.3f}', transform=axes[0].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Energy vs input norm
    input_norms = np.linalg.norm(samples, axis=1)
    axes[1].scatter(energies, input_norms, alpha=0.3, s=5, color='purple')
    axes[1].set_xlabel('Energy', fontsize=12)
    axes[1].set_ylabel('Input L2 Norm', fontsize=12)
    axes[1].set_title('Energy vs Input Magnitude', fontsize=13)
    axes[1].grid(alpha=0.3)
    
    corr_norm = np.corrcoef(energies, input_norms)[0, 1]
    axes[1].text(0.05, 0.95, f'Corr: {corr_norm:.3f}', transform=axes[1].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Energy vs feature variance
    feature_vars = np.var(samples, axis=1)
    axes[2].scatter(energies, feature_vars, alpha=0.3, s=5, color='orange')
    axes[2].set_xlabel('Energy', fontsize=12)
    axes[2].set_ylabel('Feature Variance', fontsize=12)
    axes[2].set_title('Energy vs Feature Variance', fontsize=13)
    axes[2].grid(alpha=0.3)
    
    corr_var = np.corrcoef(energies, feature_vars)[0, 1]
    axes[2].text(0.05, 0.95, f'Corr: {corr_var:.3f}', transform=axes[2].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_outliers(core_samples, tail_samples, output_path):
    """Analyze if tail jets have more outlier features (beyond [-4, 4])."""
    
    bounds = (-4, 4)
    
    core_outliers = np.sum((core_samples < bounds[0]) | (core_samples > bounds[1]), axis=1)
    tail_outliers = np.sum((tail_samples < bounds[0]) | (tail_samples > bounds[1]), axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution of outlier counts
    axes[0].hist(core_outliers, bins=50, alpha=0.6, label='Core', density=True, color='green')
    axes[0].hist(tail_outliers, bins=50, alpha=0.6, label='Tail', density=True, color='red')
    axes[0].set_xlabel('Number of Outlier Features (outside [-4, 4])', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Outlier Feature Count Distribution', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Summary statistics
    stats_text = f"Core: mean={core_outliers.mean():.2f}, median={np.median(core_outliers):.0f}\n"
    stats_text += f"Tail: mean={tail_outliers.mean():.2f}, median={np.median(tail_outliers):.0f}"
    axes[0].text(0.98, 0.97, stats_text, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Feature-wise outlier frequency
    core_outlier_freq = np.mean((core_samples < bounds[0]) | (core_samples > bounds[1]), axis=0)
    tail_outlier_freq = np.mean((tail_samples < bounds[0]) | (tail_samples > bounds[1]), axis=0)
    
    axes[1].plot(core_outlier_freq, label='Core', alpha=0.7, linewidth=1.5, color='green')
    axes[1].plot(tail_outlier_freq, label='Tail', alpha=0.7, linewidth=1.5, color='red')
    axes[1].set_xlabel('Feature Index', fontsize=12)
    axes[1].set_ylabel('Outlier Frequency', fontsize=12)
    axes[1].set_title('Per-Feature Outlier Frequency', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def main():
    print("="*80)
    print("Energy Tail Analysis")
    print("="*80)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Load model and data
    model, val_dataset, wnae_params = load_model_and_data(
        CHECKPOINT_PATH, MODEL_NAME, DATASET_CONFIG, WNAE_PRESET
    )
    
    # 2. Compute positive energies
    pos_energies, pos_samples = compute_energies_batched(model, val_dataset, BATCH_SIZE)
    
    # 3. Get negative samples and energies
    neg_energies, neg_samples = get_negative_samples_and_energies(model)
    
    # 4. Identify tail and core jets
    jet_groups = identify_tail_and_core_jets(
        pos_energies, pos_samples, TAIL_PERCENTILES, CORE_PERCENTILE
    )
    
    core_samples = jet_groups['core']['samples']
    core_energies = jet_groups['core']['energies']
    
    # Use 90th percentile tail for detailed analysis
    tail_samples = jet_groups['tail_90']['samples']
    tail_energies = jet_groups['tail_90']['energies']
    
    # 5. Feature-level analysis
    feature_stats = analyze_feature_differences(
        core_samples, tail_samples, pos_samples.shape[1]
    )
    
    # 6. Reconstruction analysis
    print("\nComputing reconstruction errors...")
    core_mse_sample, core_mse_feat = compute_reconstruction_errors(model, core_samples)
    tail_mse_sample, tail_mse_feat = compute_reconstruction_errors(model, tail_samples)
    
    print(f"Core MSE: {core_mse_sample.mean():.4f} ± {core_mse_sample.std():.4f}")
    print(f"Tail MSE: {tail_mse_sample.mean():.4f} ± {tail_mse_sample.std():.4f}")
    
    # 7. Latent space analysis
    print("\nComputing latent representations...")
    core_latent = compute_latent_codes(model, core_samples)
    tail_latent = compute_latent_codes(model, tail_samples)
    
    # 8. Generate all plots
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    # Energy distributions
    tail_thresholds = {
        f'{p}th percentile': jet_groups[f'tail_{p}']['threshold'] 
        for p in TAIL_PERCENTILES
    }
    plot_energy_distributions(
        pos_energies, neg_energies, tail_thresholds,
        os.path.join(OUTPUT_DIR, 'energy_distributions.png')
    )
    
    # Feature distributions
    plot_feature_distributions(
        core_samples, tail_samples, feature_stats,
        os.path.join(OUTPUT_DIR, 'tail_vs_core_features.png')
    )
    
    # Feature importance
    plot_feature_importance(
        feature_stats,
        os.path.join(OUTPUT_DIR, 'feature_importance.png')
    )
    
    # Reconstruction comparison
    plot_reconstruction_comparison(
        core_mse_feat, tail_mse_feat,
        os.path.join(OUTPUT_DIR, 'reconstruction_comparison.png')
    )
    
    # Latent space
    plot_latent_space(
        core_latent, tail_latent,
        os.path.join(OUTPUT_DIR, 'latent_space_pca.png'),
        method='pca'
    )
    
    # Energy correlations
    all_mse_sample, _ = compute_reconstruction_errors(model, pos_samples)
    plot_energy_correlations(
        pos_energies, pos_samples, all_mse_sample,
        os.path.join(OUTPUT_DIR, 'energy_correlations.png')
    )
    
    # Feature outliers
    plot_feature_outliers(
        core_samples, tail_samples,
        os.path.join(OUTPUT_DIR, 'feature_outliers.png')
    )
    
    # 9. Save data
    print("\n" + "="*80)
    print("Saving data...")
    print("="*80)
    
    np.save(os.path.join(DATA_DIR, 'core_samples.npy'), core_samples)
    np.save(os.path.join(DATA_DIR, 'tail_samples.npy'), tail_samples)
    np.save(os.path.join(DATA_DIR, 'core_energies.npy'), core_energies)
    np.save(os.path.join(DATA_DIR, 'tail_energies.npy'), tail_energies)
    np.save(os.path.join(DATA_DIR, 'pos_energies.npy'), pos_energies)
    np.save(os.path.join(DATA_DIR, 'neg_energies.npy'), neg_energies)
    
    # Save statistics
    statistics = {
        'model_name': MODEL_NAME,
        'wnae_preset': WNAE_PRESET,
        'num_validation_samples': len(pos_samples),
        'num_buffer_samples': len(neg_samples) if neg_samples is not None else 0,
        'pos_energy_stats': {
            'mean': float(pos_energies.mean()),
            'std': float(pos_energies.std()),
            'min': float(pos_energies.min()),
            'max': float(pos_energies.max()),
            'median': float(np.median(pos_energies))
        },
        'neg_energy_stats': {
            'mean': float(neg_energies.mean()) if neg_energies is not None else None,
            'std': float(neg_energies.std()) if neg_energies is not None else None,
            'min': float(neg_energies.min()) if neg_energies is not None else None,
            'max': float(neg_energies.max()) if neg_energies is not None else None,
            'median': float(np.median(neg_energies)) if neg_energies is not None else None
        },
        'core_stats': {
            'count': int(len(core_samples)),
            'mean_energy': float(core_energies.mean()),
            'mean_mse': float(core_mse_sample.mean())
        },
        'tail_90_stats': {
            'count': int(len(tail_samples)),
            'mean_energy': float(tail_energies.mean()),
            'mean_mse': float(tail_mse_sample.mean())
        },
        'top_10_differentiating_features': [
            {
                'feature_idx': int(f['feature_idx']),
                'ks_stat': float(f['ks_stat']),
                'mean_diff': float(f['mean_diff']),
                'core_mean': float(f['core_mean']),
                'tail_mean': float(f['tail_mean'])
            }
            for f in feature_stats[:10]
        ]
    }
    
    with open(os.path.join(DATA_DIR, 'statistics.json'), 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"Saved statistics to {os.path.join(DATA_DIR, 'statistics.json')}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print(f"Data saved to: {DATA_DIR}")
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print(f"\n1. ENERGY SEPARATION:")
    print(f"   - Positive (real) mean: {pos_energies.mean():.4f}")
    print(f"   - Negative (MCMC) mean: {neg_energies.mean():.4f}" if neg_energies is not None else "   - No negative samples")
    print(f"   - E+ > E-: {pos_energies.mean() > neg_energies.mean() if neg_energies is not None else 'N/A'}")
    
    print(f"\n2. RECONSTRUCTION QUALITY:")
    print(f"   - Core jets MSE: {core_mse_sample.mean():.4f}")
    print(f"   - Tail jets MSE: {tail_mse_sample.mean():.4f}")
    print(f"   - Tail worse by: {(tail_mse_sample.mean() / core_mse_sample.mean() - 1) * 100:.1f}%")
    
    print(f"\n3. TOP 5 DIFFERENTIATING FEATURES:")
    for i, feat in enumerate(feature_stats[:5], 1):
        print(f"   {i}. Feature {feat['feature_idx']}: KS={feat['ks_stat']:.3f}, "
              f"Δμ={feat['mean_diff']:.3f}")
    
    print(f"\n4. OUTLIER ANALYSIS:")
    core_outlier_count = np.sum((core_samples < -4) | (core_samples > 4), axis=1).mean()
    tail_outlier_count = np.sum((tail_samples < -4) | (tail_samples > 4), axis=1).mean()
    print(f"   - Core jets: {core_outlier_count:.2f} features outside [-4, 4] on average")
    print(f"   - Tail jets: {tail_outlier_count:.2f} features outside [-4, 4] on average")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
