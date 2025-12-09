# Baseline Taggers

Simple statistical baseline methods for anomaly detection, used for comparison with WNAE models.

## Gaussian Baseline Tagger

Uses the L2 norm squared (||x||²) of scaled jet features as an anomaly score. 

### Methodology

Since QCD features are preprocessed to have zero mean and unit variance, under a Gaussian assumption N(0, I), the log-likelihood is:

```
log p(x) = -0.5 * ||x||² - 0.5 * N * log(2π)
```

The anomaly score is simply ||x||², where higher values indicate more anomalous jets (farther from the typical QCD distribution).

### Usage

Basic usage with default settings:
```bash
python baseline_taggers/gaussian_baseline.py
```

With custom configuration:
```bash
python baseline_taggers/gaussian_baseline.py \
  --config data/dataset_config.json \
  --input-dim 256 \
  --max-jets 20000 \
  --output-dir baseline_taggers/results
```

With PCA-reduced features:
```bash
python baseline_taggers/gaussian_baseline.py \
  --input-dim 8 \
  --pca distance_studies/pca_output/components_std.npy \
  --output-dir baseline_taggers/results_pca8
```

### Arguments

- `--config`: Path to dataset configuration JSON (default: `data/dataset_config.json`)
- `--input-dim`: Number of input features (default: 256)
- `--max-jets`: Maximum jets per process (default: 20000)
- `--batch-size`: Batch size for loading (default: 4096)
- `--output-dir`: Directory for results (default: `baseline_taggers/results`)
- `--signals`: Signal processes to evaluate (default: GluGluHto2B SVJ TTto4Q Yto4Q)
- `--pca`: Optional PCA components file for dimensionality reduction

### Outputs

Results are saved to the specified output directory:

1. **`roc_curves.png`**: ROC curves for all signals vs QCD background
2. **`score_distributions.png`**: Histogram comparison of anomaly scores
3. **`results.json`**: Numerical results including AUC scores and ROC coordinates

### Example Results

Expected AUC ranges (will vary based on input dimensions and signals):
- Simple signals (e.g., resonances): AUC ~ 0.7-0.9
- Complex signals (e.g., TTto4Q): AUC ~ 0.5-0.7
- This provides a baseline for evaluating WNAE improvements

### Comparison with WNAE

The Gaussian baseline represents the simplest possible anomaly detector. WNAE models should significantly outperform this baseline by:
1. Learning nonlinear feature correlations
2. Capturing complex manifold structure of QCD jets
3. Using optimal transport distances for better discrimination
