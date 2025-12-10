# WNAutoencoder

## Setup
```bash
git clone git@github.com:Run3-anomaly-tagging/WNAutoencoder.git
cd WNAutoencoder
git clone ssh://git@gitlab.cern.ch:7999/cms-analysis/mlg/mlg-24-002/wnae.git wnae_root
python3 -m venv venv
echo "export PYTHONPATH=\"\${PYTHONPATH:+\$PYTHONPATH:}$(pwd):$(pwd)/wnae_root\"" >> venv/bin/activate
source venv/bin/activate
pip install -r requirements.txt
```

Test installation:
```bash
python -c "from wnae import WNAE"
```

## Usage

### Training & Evaluation

**Edit `MODEL_NAME` in `train.py` before running** (see `model_config/model_registry.py` for available models):

```bash
# Train model (auto-evaluates at end)
python train.py

# Evaluate existing checkpoint
python evaluate_wnae.py --checkpoint models/MODEL/checkpoint.pth --model-name MODEL_NAME
```

Parameter presets defined in `model_config/model_config.py`.

### Data Preprocessing (`data/`)

Standardize QCD features (zero mean, unit variance) and apply to signals:

```bash
# Interactive menu
python data/preprocessing.py

# CLI mode
python data/preprocessing.py --merge-qcd-ht --scale-qcd --scale-signals
python data/preprocessing.py --inspect data/file.h5
```

**Outputs:** `data/*_scaled.h5`, `results/scaled_feature_histograms/`

### Distance Studies (`distance_studies/`)

Analyze Wasserstein/Sinkhorn distance metrics:

```bash
# Gaussian validation test
python distance_studies/distance_resolution.py --dim 4 --batch 4096

# Cross-dimensional comparison
python distance_studies/distance_study.py --compute --dims 8 16 32 64

# High-dimensional (sliced Wasserstein)
python distance_studies/sliced_wasserstein_study.py --dim 256

# MCMC buffer diagnostics (requires trained checkpoint)
python distance_studies/distance_study_sws.py --model MODEL_NAME --cfg CFG1
```

**Outputs:** `results/distance_study/`

### Gaussian Studies (`gaussian_studies/`)

Baseline taggers and Gaussian fit tests:

```bash
# Gaussian likelihood baseline tagger
python gaussian_studies/gaussian_likelihood.py --n-train 100000 --n-test 50000

# Test Gaussian fit quality
python gaussian_studies/gaussian_fit_test.py --process QCD --n-samples 50000

# L2 norm baseline
python gaussian_studies/gaussian_baseline.py --max-jets 20000

# Correlation analysis (takes a while to run)
python gaussian_studies/correlation_analysis.py --batch_size 10000 --subsample_dcor 5000
```

**Outputs:** `results/gaussian_studies/`