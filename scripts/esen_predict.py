from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import yaml
from ase.io import read
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fairchem.core import FAIRChemCalculator

"""Usage:
uv run scripts/esen_predict.py --checkpoint /scratch/aburger/checkpoint/uma/202512-1802-3934-ed7c/checkpoints
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run eSEN / UMA energy predictions on fixed XYZ test sets "
            " and save CSV files."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained eSEN / UMA checkpoint (.pt).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="esen-predictions",
        help="Wandb project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        help="Wandb run name. If not provided, will use checkpoint name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    datasets = {
        "qm7": Path("data/all/8020/qm7_validation.xyz"),
        "amino_acids": Path("data/amino_acids.xyz"),
        "alcohols": Path("data/alcohols.xyz"),
        "alkanes": Path("data/alkanes.xyz"),
        "pubchem": Path("data/pubchem.xyz"),
        "stretching_mol": Path("data/stretching_mol.xyz"),
    }

    checkpoint_path = args.checkpoint

    # if checkpoint path is a directory, find latest step checkpoint
    # /scratch/aburger/checkpoint/uma/202512-1717-3450-f34f/checkpoints
    if checkpoint_path.is_dir():
        # get latest checkpoint dir
        checkpoint_dir = list(checkpoint_path.glob("step_*"))[-1]
        # get checkpoint file
        checkpoint_path = list(checkpoint_dir.glob("*.pt"))[-1]
    else:
        checkpoint_dir = checkpoint_path.parent
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist")

    calc = FAIRChemCalculator.from_model_checkpoint(
        name_or_path=str(checkpoint_path),
        task_name=None,  # auto-detect from checkpoint (e.g., "custom" for fine-tuned models)
        device=None,
    )
    
    # get training config
    with open(checkpoint_dir / "resume.yaml", 'r') as f:
        training_config = yaml.safe_load(f)
    
    # Initialize wandb
    run_name = args.wandb_run_name or checkpoint_path.stem
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "checkpoint_path": str(checkpoint_path),
            "training_config": training_config,
        }
    )

    # Define summary metrics with minimize summary (lower is better)
    metric_suffixes = ["mae_structure", "mse_structure", "rmse_structure", "mae_atom", "mse_atom", "rmse_atom"]
    for dataset_name in datasets.keys():
        for suffix in metric_suffixes:
            wandb.define_metric(f"{dataset_name}/{suffix}", summary="min")

    summary_str = ""
    for name, xyz_path in datasets.items():
        print(f"\n# Predicting {name}.")
        atoms_list = read(str(xyz_path), index=":")

        rows = []
        for idx, atoms in enumerate(atoms_list):
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            n_atoms = len(atoms)
            symbols = "".join(atoms.get_chemical_symbols())

            ref_energy = atoms.info.get("REF_energy", np.nan)

            rows.append(
                {
                    "index": idx,
                    "n_atoms": n_atoms,
                    "symbols": symbols,
                    "REF_energy": ref_energy,
                    "ESEN_energy": float(energy),
                }
            )

        df = pd.DataFrame(rows)

        output_path = Path("data") / f"predictions_{name}_esen.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

        # Compute metrics for this dataset
        print(f"Total rows: {len(df)}")
        valid_rows = df.dropna(subset=['REF_energy'])
        print(f"Valid rows: {len(valid_rows)}")
        if len(valid_rows) > 0:
            y_true = valid_rows['REF_energy'].values
            y_pred = valid_rows['ESEN_energy'].values

            # Per structure metrics
            mae_structure = mean_absolute_error(y_true, y_pred)
            mse_structure = mean_squared_error(y_true, y_pred)
            rmse_structure = np.sqrt(mse_structure)

            # Per atom metrics (weighted by number of atoms)
            weights = 1.0 / valid_rows['n_atoms'].values
            mae_atom = np.average(np.abs(y_pred - y_true), weights=weights)
            mse_atom = np.average((y_pred - y_true)**2, weights=weights)
            rmse_atom = np.sqrt(mse_atom)

            # Log to wandb
            metrics = {
                f"{name}/mae_structure": mae_structure,
                f"{name}/mse_structure": mse_structure,
                f"{name}/rmse_structure": rmse_structure,
                f"{name}/mae_atom": mae_atom,
                f"{name}/mse_atom": mse_atom,
                f"{name}/rmse_atom": rmse_atom,
                f"{name}/num_structures": len(valid_rows),
            }
            wandb.log(metrics, step=0)

            # Set as summary metrics explicitly
            for key, value in metrics.items():
                wandb.run.summary[key] = value
            _metrics_str = f"Metrics for {name}: MAE={mae_structure*1e3:.4f}, RMSE={rmse_structure*1e3:.4f} [mH] (per structure)"
            summary_str += _metrics_str + "\n"
        else:
            _metrics_str = f"Warning: No reference energies found for {name}, skipping metrics"
            summary_str += _metrics_str + "\n"
        print(_metrics_str)

    print("")
    print(summary_str)

    print("\nAll predictions saved!")
    wandb.finish()

if __name__ == "__main__":
    main()
