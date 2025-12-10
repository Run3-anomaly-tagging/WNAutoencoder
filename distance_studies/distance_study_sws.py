"""
Compare distances between real data samples and MCMC negative samples.

Loads a trained model checkpoint, extracts buffer (negative samples from MCMC),
and compares distances:
- nn: negative-negative (within buffer)
- pp: positive-positive (real data samples)
- np: negative-positive (buffer vs real data)

Useful for debugging MCMC sampling quality.
"""
import os
import sys
import json
import torch
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from torch.utils.data import DataLoader, RandomSampler
from utils.jet_dataset import JetDataset
from model_config.model_registry import MODEL_REGISTRY
from model_config.model_config import WNAE_PARAM_PRESETS
from wnae import WNAE

def sample_batch(dataset: JetDataset, batch_size: int):
    """Sample a random batch from dataset without replacement."""
    loader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset, replacement=False), pin_memory=True)
    batch = next(iter(loader))[0]
    return batch

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare sliced Wasserstein distances: buffer vs real data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python distance_studies/distance_study_sws.py --model deep_bottleneck_qcd --cfg CFG1 --batch 4096
        """
    )
    parser.add_argument("--model", type=str, default="deep_bottleneck_qcd", help="Model name from registry")
    parser.add_argument("--cfg", type=str, default="CFG1", help="WNAE parameter preset")
    parser.add_argument("--loss", type=str, default="ae", help="Loss function suffix in checkpoint path")
    parser.add_argument("--batch", type=int, default=4096, help="Batch size")
    parser.add_argument("--dataset", type=str, default="data/QCD_merged_scaled.h5", help="Dataset path")
    parser.add_argument("--output", type=str, default="results/distance_study/buffer_distances.json", help="Output JSON file")
    args = parser.parse_args()

    model_name = args.model
    cfg = args.cfg
    batch_size = args.batch
    
    wnae_params = WNAE_PARAM_PRESETS[cfg]
    model_config = MODEL_REGISTRY[model_name]
    
    model = WNAE(
        encoder=model_config["encoder"](),
        decoder=model_config["decoder"](),
        **wnae_params
    )
    
    SAVEDIR = f"{model_config['savedir']}_{args.loss}_{cfg}"
    CHECKPOINT_PATH = os.path.join(project_root, "models", SAVEDIR, "checkpoint.pth")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)
    
    print(f"[INFO] Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=torch.device('cpu'))
    
    buffer = checkpoint["buffer"]
    final_loss = checkpoint['training_losses'][-1]
    print(f"[INFO] Final training loss: {final_loss:.4f}")
    print(f"[INFO] Buffer size: {len(buffer)}")
    
    # Sample negative-negative pairs from buffer
    batch1_n = buffer[:batch_size]
    batch2_n = buffer[-batch_size:]
    nn_distance = WNAE.compute_sliced_wasserstein_distance(batch1_n, batch2_n).item()

    # Sample positive-positive pairs from dataset
    dataset_path = os.path.join(project_root, args.dataset)
    dataset = JetDataset(dataset_path, input_dim=model_config["input_dim"])
    batch1 = sample_batch(dataset, batch_size=batch_size)
    batch2 = sample_batch(dataset, batch_size=batch_size)
    pp_distance = WNAE.compute_sliced_wasserstein_distance(batch1, batch2).item()
    
    # Sample negative-positive pair
    np_distance = WNAE.compute_sliced_wasserstein_distance(batch1, batch2_n).item()
    dataset.close()

    print()
    print("Sliced Wasserstein Distances:")
    print(f"  nn (negative-negative): {nn_distance:.4f}")
    print(f"  pp (positive-positive): {pp_distance:.4f}")
    print(f"  np (negative-positive): {np_distance:.4f}")
    print()
    print("Interpretation:")
    print(f"  - nn should be small (MCMC samples are similar)")
    print(f"  - pp should be small (real data is self-consistent)")
    print(f"  - np should be larger if model separates distributions")
    
    # Save results to JSON
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = {
        "model": model_name,
        "cfg": cfg,
        "loss": args.loss,
        "batch_size": batch_size,
        "final_training_loss": final_loss,
        "buffer_size": len(buffer),
        "distances": {
            "nn": nn_distance,
            "pp": pp_distance,
            "np": np_distance
        },
        "checkpoint": CHECKPOINT_PATH,
        "dataset": dataset_path
    }
    
    # Load existing results if file exists
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    # Generate unique key
    from datetime import datetime
    import uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = f"{model_name}_{cfg}_{timestamp}_{uuid.uuid4().hex[:6]}"
    all_results[key] = results
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[INFO] Results saved to: {output_path}")
    print(f"[INFO] Entry key: {key}")

