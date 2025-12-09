# Energy Tail Analysis

## Overview
This analysis investigates why some QCD jets end up in the high-energy tail of the positive energy distribution (E+), while negative MCMC samples concentrate in the low-energy core.

## Model Under Investigation
- **Model**: `deep_bottleneck_qcd` 
- **Configuration**: `wnae_CFG1`
- **Checkpoint**: `../models/deep_bottleneck_qcd_wnae_CFG1/checkpoint.pth`

## Key Questions
1. Do high-energy jets have outlier features (far from [-4, 4] range)?
2. Are they poorly reconstructed in specific dimensions?
3. Do they have distinct physical properties?
4. Is the tail due to model capacity limitations or data distribution?
5. Could tighter MCMC bounds prevent exploring these regions?

## Usage

### Run Full Analysis
```bash
cd energy_tail_analysis
python analyze_energy_tails.py
```

### Requirements
- Trained WNAE model checkpoint
- QCD validation dataset
- Required packages: torch, numpy, matplotlib, scipy, sklearn, tqdm

## Output Structure

### Plots (`plots/`)
- `energy_distributions.png` - E+ vs E- histograms with percentile markers
- `tail_vs_core_features.png` - Top 8 differentiating features
- `feature_importance.png` - Ranked by KS statistic and mean difference
- `reconstruction_comparison.png` - MSE per feature for core vs tail
- `latent_space_pca.png` - 2D PCA projection of latent codes
- `energy_correlations.png` - Energy vs MSE, input norm, feature variance
- `feature_outliers.png` - Analysis of features outside [-4, 4] bounds

### Data (`data/`)
- `core_samples.npy` - Jets in low-energy core (<50th percentile)
- `tail_samples.npy` - Jets in high-energy tail (>90th percentile)
- `core_energies.npy` - Energies for core jets
- `tail_energies.npy` - Energies for tail jets
- `pos_energies.npy` - All positive (real) energies
- `neg_energies.npy` - All negative (MCMC) energies
- `statistics.json` - Summary statistics and top differentiating features

## Analysis Components

### 1. Energy Distribution Analysis
Compares positive (real QCD) vs negative (MCMC) energy distributions with percentile thresholds at 80th and 90th.

### 2. Feature-Level Analysis
- Identifies features with largest distribution differences (KS test)
- Computes mean/std for core vs tail jets
- Statistical significance testing (t-test, KS test)

### 3. Reconstruction Quality
- Per-feature MSE comparison
- Identifies poorly reconstructed dimensions
- Correlates reconstruction error with energy

### 4. Latent Space Structure
- PCA visualization of core vs tail separation
- Checks if tail jets form distinct cluster

### 5. Outlier Detection
- Counts features outside MCMC bounds [-4, 4]
- Per-feature outlier frequency analysis

### 6. Correlation Studies
- Energy vs reconstruction MSE
- Energy vs input L2 norm
- Energy vs feature variance

## Key Findings Format
The script outputs:
1. Energy separation statistics (E+ vs E-)
2. Reconstruction quality comparison
3. Top differentiating features
4. Outlier analysis summary

## Notes
- Uses same train/val split as training (80/20 with seed=0)
- Batched computation to handle ~20k validation samples
- All plots use density normalization for fair comparison
- Statistics saved in JSON format for downstream analysis
