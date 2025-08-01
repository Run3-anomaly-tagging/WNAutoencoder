
import h5py
import numpy as np
import os
from typing import Optional

def count_jets_in_file(filepath: str) -> int:
    """Count number of jets in an h5 file."""
    try:
        with h5py.File(filepath, 'r') as f:
            return len(f['Jets'])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0

def load_jets_from_file(filepath: str, max_jets: Optional[int] = None) -> np.ndarray:
    """Load jets from h5 file with optional max limit."""
    with h5py.File(filepath, 'r') as f:
        jets = f['Jets'][:]
        if max_jets is not None and len(jets) > max_jets:
            # Random sampling to avoid bias
            indices = np.random.choice(len(jets), max_jets, replace=False)
            jets = jets[indices]
        return jets

def save_jets_to_file(filepath: str, jets: np.ndarray, weights: Optional[np.ndarray] = None):
    """Save jets to h5 file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('Jets', data=jets)
        if weights is not None:
            f.create_dataset('Weights', data=weights)

def extract_jet_images(jets: np.ndarray) -> np.ndarray:
    """Extract jet images from jet data."""
    return jets['jet_image']

def extract_hidden_features(jets: np.ndarray) -> np.ndarray:
    """Extract hidden neuron features from jet data."""
    return jets['hidNeurons']