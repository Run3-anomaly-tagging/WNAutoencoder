# Energy Tail Analysis

## Overview
This analysis investigates why some QCD jets end up in the high-energy tail of the positive energy distribution (E+). Negative MCMC samples concentrate in the low-energy core.

## Model Under Investigation
- **Model**: `deep_bottleneck_qcd` 
- **Configuration**: `wnae_CFG1`
- **Checkpoint**: `../models/deep_bottleneck_qcd_wnae_CFG1/checkpoint.pth`

## Usage

### Run Full Analysis
```bash
cd energy_tail_analysis
python analyze_energy_tails.py
```

### Requirements
- Trained WNAE model checkpoint
- QCD validation dataset
