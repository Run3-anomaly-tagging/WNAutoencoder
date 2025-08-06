# WNAutoencoder
```
git clone git@github.com:Run3-anomaly-tagging/WNAutoencoder.git
cd WNAutoencoder
git clone ssh://git@gitlab.cern.ch:7999/cms-analysis/mlg/mlg-24-002/wnae.git wnae_root
python3 -m venv venv
echo "export PYTHONPATH=\"\${PYTHONPATH:+\$PYTHONPATH:}$(pwd)/wnae_root\"" >> venv/bin/activate
source venv/bin/activate
pip install -r requirements.txt
```

To test
```
python -c "from wnae import WNAE"
```