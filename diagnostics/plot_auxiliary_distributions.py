"""
Diagnostic script to visualize auxiliary variable distributions from multiple datasets.
Loads data through JetDataset to ensure we're seeing the actual scaled values used in training.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import roc_curve, auc

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils.jet_dataset import JetDataset


def plot_auxiliary_distributions(
    datasets_config,
    aux_keys,
    input_dim=256,
    max_jets=50000,
    savedir=None,
    aux_quantile_transformer=None
):
    """
    Plot distributions of auxiliary variables from multiple datasets.
    Shows three types: raw values, discriminants (ratios), and quantile-transformed.
    
    Args:
        datasets_config: List of tuples (name, filepath)
        aux_keys: List of auxiliary feature keys to plot
        input_dim: Number of base features (before auxiliary)
        max_jets: Maximum number of jets to load per dataset
        savedir: Directory to save plots (default: diagnostics/plots)
        aux_quantile_transformer: Path to quantile transformer or fitted object
    """
    if savedir is None:
        savedir = os.path.join(script_dir, "plots")
    os.makedirs(savedir, exist_ok=True)
    
    datasets_raw = {}
    datasets_discriminants = {}
    datasets_transformed = {}
    
    for name, filepath in datasets_config:
        print(f"Loading {name} from {filepath}...")
        
        indices = np.arange(max_jets) if max_jets else None
        
        dataset = JetDataset(
            filepath=filepath,
            input_dim=input_dim,
            aux_keys=aux_keys,
            indices=indices,
            aux_quantile_transformer=aux_quantile_transformer
        )
        
        # Extract transformed auxiliary values (last aux_dim dimensions)
        aux_dim = len(aux_keys)
        all_aux_transformed = []
        for i in range(len(dataset)):
            features, _ = dataset[i]
            aux_values = features[-aux_dim:].numpy()
            all_aux_transformed.append(aux_values)
        
        datasets_transformed[name] = np.array(all_aux_transformed)
        
        import h5py
        all_aux_raw = []
        with h5py.File(filepath, 'r') as f:
            jets = f['Jets']
            for aux_key in aux_keys:
                if aux_key in jets.dtype.names:
                    aux_data = jets[aux_key][:max_jets] if max_jets else jets[aux_key][:]
                    all_aux_raw.append(aux_data)
                else:
                    print(f"    Warning: {aux_key} not found in {filepath}")
                    print(f"    Available fields: {jets.dtype.names}")
        
        if len(all_aux_raw) > 0:
            aux_raw_matrix = np.array(all_aux_raw).T  # Shape: (n_jets, n_aux)
            datasets_raw[name] = aux_raw_matrix
            
            # Compute discriminants: disc_i = x_i / sum(x_j)
            aux_sum = aux_raw_matrix.sum(axis=1, keepdims=True) + 1e-8
            datasets_discriminants[name] = aux_raw_matrix / aux_sum
        else:
            datasets_raw[name] = np.zeros((len(datasets_transformed[name]), len(aux_keys)))
            datasets_discriminants[name] = np.zeros((len(datasets_transformed[name]), len(aux_keys)))
            print(f"    Warning: No raw auxiliary data found, using zeros")
        
        print(f"  Loaded {len(datasets_transformed[name])} jets")
        print(f"    Shapes - Raw: {datasets_raw[name].shape}, Discriminants: {datasets_discriminants[name].shape}, Transformed: {datasets_transformed[name].shape}")
    
    print("\nPlotting RAW auxiliary distributions")
    for aux_idx, aux_key in enumerate(aux_keys):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_values = np.concatenate([aux_data[:, aux_idx] for aux_data in datasets_raw.values()])
        vmin, vmax = np.percentile(all_values, [0.5, 99.5])
        
        for name, aux_data in datasets_raw.items():
            values = aux_data[:, aux_idx]
            ax.hist(
                values,
                bins=100,
                alpha=0.6,
                label=f"{name} (mean={values.mean():.3f}, std={values.std():.3f})",
                density=True,
                range=(vmin, vmax)
            )
        
        ax.set_xlabel(f"{aux_key} (raw)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Auxiliary Variable (Raw): {aux_key}", fontsize=14, fontweight='bold')
        ax.set_xlim(vmin, vmax)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plot_filename = f"aux_{aux_key}_raw_distribution.png"
        plot_path = os.path.join(savedir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()
    

    print("\nPlotting DISCRIMINANT distributions")
    for aux_idx, aux_key in enumerate(aux_keys):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, aux_data in datasets_discriminants.items():
            values = aux_data[:, aux_idx]
            ax.hist(
                values,
                bins=100,
                alpha=0.6,
                label=f"{name} (mean={values.mean():.3f}, std={values.std():.3f})",
                density=True,
                range=(0, 1)
            )
        
        ax.set_xlabel(f"{aux_key} (discriminant: x_i / Σx_j)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Auxiliary Discriminant: {aux_key}", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plot_filename = f"aux_{aux_key}_discriminant_distribution.png"
        plot_path = os.path.join(savedir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()
    
    print("\nPlotting QUANTILE-TRANSFORMED distributions")
    for aux_idx, aux_key in enumerate(aux_keys):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, aux_data in datasets_transformed.items():
            values = aux_data[:, aux_idx]
            ax.hist(
                values,
                bins=100,
                alpha=0.6,
                label=f"{name} (mean={values.mean():.3f}, std={values.std():.3f})",
                density=True,
                range=(-5, 5)
            )
        
        ax.set_xlabel(f"{aux_key} (quantile → Gaussian)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Auxiliary Variable (Gaussianized): {aux_key}", fontsize=14, fontweight='bold')
        ax.set_xlim(-5, 5)

        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plot_filename = f"aux_{aux_key}_gaussianized_distribution.png"
        plot_path = os.path.join(savedir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()
    
    n_aux = len(aux_keys)
    
    # Combined: RAW
    fig, axes = plt.subplots(1, n_aux, figsize=(6*n_aux, 5))
    if n_aux == 1:
        axes = [axes]
    
    for aux_idx, (aux_key, ax) in enumerate(zip(aux_keys, axes)):
        all_values = np.concatenate([aux_data[:, aux_idx] for aux_data in datasets_raw.values()])
        vmin, vmax = np.percentile(all_values, [0.5, 99.5])
        
        for name, aux_data in datasets_raw.items():
            values = aux_data[:, aux_idx]
            ax.hist(values, bins=100, alpha=0.6, label=name, density=True, range=(vmin, vmax))
        
        ax.set_xlabel(f"{aux_key} (raw)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{aux_key}", fontsize=12, fontweight='bold')
        ax.set_xlim(vmin, vmax)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Auxiliary Variables (Raw) - All Datasets", fontsize=14, fontweight='bold', y=1.02)
    plot_path = os.path.join(savedir, "aux_all_variables_raw_combined.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    # Combined: DISCRIMINANTS
    fig, axes = plt.subplots(1, n_aux, figsize=(6*n_aux, 5))
    if n_aux == 1:
        axes = [axes]
    
    for aux_idx, (aux_key, ax) in enumerate(zip(aux_keys, axes)):
        for name, aux_data in datasets_discriminants.items():
            values = aux_data[:, aux_idx]
            ax.hist(values, bins=100, alpha=0.6, label=name, density=True, range=(0, 1))
        
        ax.set_xlabel(f"{aux_key} (disc)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{aux_key}", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Auxiliary Discriminants (x_i / Sum(x_j) - All Datasets", fontsize=14, fontweight='bold', y=1.02)
    plot_path = os.path.join(savedir, "aux_all_variables_discriminant_combined.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    # Combined: TRANSFORMED
    fig, axes = plt.subplots(1, n_aux, figsize=(6*n_aux, 5))
    if n_aux == 1:
        axes = [axes]
    
    for aux_idx, (aux_key, ax) in enumerate(zip(aux_keys, axes)):
        all_values = np.concatenate([aux_data[:, aux_idx] for aux_data in datasets_transformed.values()])
        vmin, vmax = np.percentile(all_values, [0.5, 99.5])
        
        for name, aux_data in datasets_transformed.items():
            values = aux_data[:, aux_idx]
            ax.hist(values, bins=100, alpha=0.6, label=name, density=True, range=(vmin, vmax))
        
        ax.set_xlabel(f"{aux_key} (Gaussian)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{aux_key}", fontsize=12, fontweight='bold')
        ax.set_xlim(vmin, vmax)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Auxiliary Variables (Gaussianized) - All Datasets", fontsize=14, fontweight='bold', y=1.02)
    plot_path = os.path.join(savedir, "aux_all_variables_gaussianized_combined.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    # ROC curves for sanity check
    print("\nComputing ROC curves for QCD vs Top_bqq separation...")
    
    if len(datasets_config) == 2 and datasets_config[0][0] == "QCD" and datasets_config[1][0] == "Top_bqq":
        qcd_name, top_name = datasets_config[0][0], datasets_config[1][0]
        
        # Labels: QCD=0 (background), Top=1 (signal)
        qcd_labels = np.zeros(len(datasets_raw[qcd_name]))
        top_labels = np.ones(len(datasets_raw[top_name]))
        all_labels = np.concatenate([qcd_labels, top_labels])
        
        # Plot ROC for each aux variable in 3 representations
        for aux_idx, aux_key in enumerate(aux_keys):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # 1) Raw auxiliary values
            qcd_raw = datasets_raw[qcd_name][:, aux_idx]
            top_raw = datasets_raw[top_name][:, aux_idx]
            all_raw = np.concatenate([qcd_raw, top_raw])
            
            fpr_raw, tpr_raw, _ = roc_curve(all_labels, all_raw)
            auc_raw = auc(fpr_raw, tpr_raw)
            
            axes[0].plot(fpr_raw, tpr_raw, lw=2, label=f'AUC = {auc_raw:.3f}')
            axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
            axes[0].set_xlabel('False Positive Rate', fontsize=11)
            axes[0].set_ylabel('True Positive Rate', fontsize=11)
            axes[0].set_title(f'{aux_key} (Raw)', fontsize=12, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # 2) Discriminants
            qcd_disc = datasets_discriminants[qcd_name][:, aux_idx]
            top_disc = datasets_discriminants[top_name][:, aux_idx]
            all_disc = np.concatenate([qcd_disc, top_disc])
            
            fpr_disc, tpr_disc, _ = roc_curve(all_labels, all_disc)
            auc_disc = auc(fpr_disc, tpr_disc)
            
            axes[1].plot(fpr_disc, tpr_disc, lw=2, label=f'AUC = {auc_disc:.3f}')
            axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
            axes[1].set_xlabel('False Positive Rate', fontsize=11)
            axes[1].set_ylabel('True Positive Rate', fontsize=11)
            axes[1].set_title(f'{aux_key} (Discriminant)', fontsize=12, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            
            # 3) Quantile-transformed
            qcd_trans = datasets_transformed[qcd_name][:, aux_idx]
            top_trans = datasets_transformed[top_name][:, aux_idx]
            all_trans = np.concatenate([qcd_trans, top_trans])
            
            fpr_trans, tpr_trans, _ = roc_curve(all_labels, all_trans)
            auc_trans = auc(fpr_trans, tpr_trans)
            
            axes[2].plot(fpr_trans, tpr_trans, lw=2, label=f'AUC = {auc_trans:.3f}')
            axes[2].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
            axes[2].set_xlabel('False Positive Rate', fontsize=11)
            axes[2].set_ylabel('True Positive Rate', fontsize=11)
            axes[2].set_title(f'{aux_key} (Gaussianized)', fontsize=12, fontweight='bold')
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'ROC Curves: QCD vs Top_bqq Separation ({aux_key})', fontsize=14, fontweight='bold')
            plot_path = os.path.join(savedir, f"roc_{aux_key}_comparison.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved ROC: {plot_path}")
            print(f"  AUC - Raw: {auc_raw:.3f}, Discriminant: {auc_disc:.3f}, Gaussianized: {auc_trans:.3f}")
            plt.close()
    
    print(f"\nAll plots saved to: {savedir}")


if __name__ == "__main__":
    CONFIG_PATH = os.path.join(project_root, "data", "dataset_config.json")
    
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    DATASETS = [
        ("QCD", config["QCD"]["path"]),
        ("Top_bqq", config["Top_bqq"]["path"])
    ]
    
    AUX_KEYS = ['globalParT3_QCD', 'globalParT3_TopbWqq']
    
    INPUT_DIM = 256
    MAX_JETS = 50000
    QUANTILE_TRANSFORMER_PATH = os.path.join(project_root, "data", "aux_quantile_transformer.pkl")
    
    if not os.path.exists(QUANTILE_TRANSFORMER_PATH):
        print("=" * 80)
        print("ERROR: Quantile transformer not found!")
        print("=" * 80)
        print(f"Expected location: {QUANTILE_TRANSFORMER_PATH}")
        print("\nPlease run the following command first to fit the transformer:")
        print("  python utils/fit_aux_quantile_transformer.py")
        print("\nThis will analyze the training data and create the transformer.")
        print("=" * 80)
        sys.exit(1)
    
    print("=" * 80)
    print("AUXILIARY VARIABLE DISTRIBUTION DIAGNOSTIC")
    print("=" * 80)
    print(f"Datasets: {[name for name, _ in DATASETS]}")
    print(f"Auxiliary keys: {AUX_KEYS}")
    print(f"Quantile transformer: {QUANTILE_TRANSFORMER_PATH}")
    print("=" * 80)
    print()
    
    plot_auxiliary_distributions(
        datasets_config=DATASETS,
        aux_keys=AUX_KEYS,
        input_dim=INPUT_DIM,
        max_jets=MAX_JETS,
        aux_quantile_transformer=QUANTILE_TRANSFORMER_PATH
    )
