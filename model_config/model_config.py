import json
import os
from model_config.model_registry import MODEL_REGISTRY

WNAE_PARAM_PRESETS = {
    "DEFAULT_WNAE_PARAMS" : {
        "sampling": "pcd",
        "n_steps":10,
        "noise":0.05,
        "step_size":None,
        "temperature": 1.0,
        "bounds": (-4.,4.),
        "mh": False,
        "initial_distribution": "gaussian",
        "replay": True,
        "replay_ratio": 0.95
    },
    "TUTORIAL_WNAE_PARAMS" : {
        "sampling": "pcd",
        "n_steps": 10,
        "step_size": None,
        "noise": 0.2,
        "temperature": 0.05,
        "bounds": (-4, 4),#In tutorial, this is (-3,3)
        "mh": False,
        "initial_distribution": "gaussian",
        "replay": True,
        "replay_ratio": 0.95,
        "buffer_size": 10000,
        "distance": "sinkhorn"
    }
}

DEFAULT_TRAINING_PARAMS = {
    "BATCH_SIZE": 4096,
    "NUM_SAMPLES": 2**17,
    "LEARNING_RATE": 1e-3,
    "N_EPOCHS": 1
}

def load_dataset_path(process_name: str, dataset_config_file: str ):
    """Fetch dataset path from a JSON config using the process name."""
    if not os.path.exists(dataset_config_file):
        raise FileNotFoundError(f"Dataset config file not found: {dataset_config_file}")
    with open(dataset_config_file, "r") as f:
        data_cfg = json.load(f)
    if process_name not in data_cfg:
        raise KeyError(f"Process '{process_name}' not found in {dataset_config_file}")
    return data_cfg[process_name]["path"]


def build_training_config(model_name: str, dataset_config_file: str ):
    """
    Returns a ready-to-use training config dict.
    Combines model details, dataset path, WNAE + training parameters.
    """
    model_config = MODEL_REGISTRY[model_name]
    process_name = model_config["process"]
    data_path = load_dataset_path(process_name, dataset_config_file)

    training_config = {
        "MODEL_NAME": model_name,
        "MODEL_CONFIG": model_config,
        "DATA_PATH": data_path,
        "INPUT_DIM": model_config["input_dim"],
        "WNAE_PARAMS": DEFAULT_WNAE_PARAMS.copy(),
        **DEFAULT_TRAINING_PARAMS
    }
    return training_config
