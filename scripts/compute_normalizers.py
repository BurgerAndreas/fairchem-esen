#!/usr/bin/env python3
"""
Script to compute normalizers from training data.
Usage: python scripts/compute_normalizers.py --config configs/uma/training_release/esen_sm_direct_lmbm.yaml
"""

import argparse
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from fairchem.core.datasets.mt_concat_dataset import create_concat_dataset
from fairchem.core.modules.normalization.normalizer import fit_normalizers
from fairchem.core.common.utils import save_checkpoint

logging.basicConfig(level=logging.INFO)


def compute_normalizers(config_path: str, output_path: str = "./normalizers.pt", num_batches: int = None):
    """Compute normalizers from training dataset."""
    
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Create training dataset (same as in your config)
    train_dataset = create_concat_dataset(
        dataset_configs=cfg.train_dataset.dataset_configs,
        combined_dataset_config=cfg.train_dataset.combined_dataset_config
    )
    
    logging.info(f"Dataset size: {len(train_dataset)}")
    
    # Fit normalizers
    # For energy task, we typically want mean=0.0 and compute RMSD from data
    normalizers = fit_normalizers(
        targets=["energy"],  # Change if you have other targets (e.g., "forces")
        dataset=train_dataset,
        batch_size=32,  # Adjust as needed
        num_batches=num_batches,  # None = use all data, or set to limit (e.g., 100)
        override_values={
            "energy": {"mean": 0.0}  # Force mean to 0, compute RMSD from data
        },
        num_workers=4,  # Adjust based on your system
        shuffle=True,
        seed=0,
    )
    
    # Print computed values
    for target, normalizer in normalizers.items():
        mean = normalizer.mean.item()
        rmsd = normalizer.rmsd.item()
        logging.info(f"{target}: mean={mean:.6f}, rmsd={rmsd:.6f}")
    
    # Save normalizers
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(normalizers, output_path.parent, output_path.name)
    logging.info(f"Normalizers saved to: {output_path}")
    
    # Print values to use in config
    print("\n" + "="*60)
    print("Add these values to your config:")
    print("="*60)
    for target, normalizer in normalizers.items():
        mean = normalizer.mean.item()
        rmsd = normalizer.rmsd.item()
        print(f"normalizer_rmsd: {rmsd:.6f}  # for {target}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute normalizers from training data")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./normalizers.pt",
        help="Output path for normalizers checkpoint"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Number of batches to use for fitting (None = use all)"
    )
    args = parser.parse_args()
    
    compute_normalizers(args.config, args.output, args.num_batches)

