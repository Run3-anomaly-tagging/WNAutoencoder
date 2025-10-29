# WNAutoencoder
```
git clone git@github.com:Run3-anomaly-tagging/WNAutoencoder.git
cd WNAutoencoder
git clone ssh://git@gitlab.cern.ch:7999/cms-analysis/mlg/mlg-24-002/wnae.git wnae_root
python3 -m venv venv
echo "export PYTHONPATH=\"\${PYTHONPATH:+\$PYTHONPATH:}$(pwd):$(pwd)/wnae_root\"" >> venv/bin/activate
source venv/bin/activate
pip install -r requirements.txt
```

To test the installation
```
python -c "from wnae import WNAE"
```

### To train and evaluate the model

Before running the training or evaluation scripts, make sure to set the `MODEL_NAME` variable inside the scripts (`train.py` and `evaluate_wnae.py`) to specify which architecture to use. The available model names are defined in `model_registry.py`.
```
python train.py
python evaluate_wnae.py --checkpoint path/to/checkpoint.pth --model-name MODEL_NAME --config data/dataset_config_small.json #Check other CLI options
```

Parameter sets for WNAE models are defined in `model_config/model_config.py`

### To preprocess input files

The preprocessing script rescales QCD jet features to zero mean and unit standard deviation, and applies the same scaling to signal jet features. There are other options for manipulating datasets as well. 
> **Note:** You need to manually edit the main function in `preprocessing.py` to select the input files and specify which scaling to apply.
```
cd data
python preprocessing.py
```