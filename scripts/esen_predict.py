from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

from fairchem.core import FAIRChemCalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run eSEN / UMA energy predictions on fixed XYZ test sets "
            "(amino_acids, alcohols, alkanes, pubchem) and save CSV files."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained eSEN / UMA checkpoint (.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint
    
    # if checkpoint path is a directory, find latest step checkpoint
    # /scratch/aburger/checkpoint/uma/202512-1717-3450-f34f/checkpoints
    if checkpoint_path.is_dir():
        # get latest checkpoint dir
        checkpoint_path = list(checkpoint_path.glob("step_*"))[-1]
        # get checkpoint file
        checkpoint_path = list(checkpoint_path.glob("*.pt"))[-1]
    
    checkpoint_path = checkpoint_path.resolve()
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist")

    calc = FAIRChemCalculator.from_model_checkpoint(
        name_or_path=str(checkpoint_path),
        task_name="omol",
        device=None,
    )

    datasets = {
        "qm7": Path("data/all/8020/qm7_validation.xyz"),
        "amino_acids": Path("data/amino_acids.xyz"),
        "alcohols": Path("data/alcohols.xyz"),
        "alkanes": Path("data/alkanes.xyz"),
        "pubchem": Path("data/pubchem.xyz"),
    }

    for name, xyz_path in datasets.items():
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

    print("\nAll predictions saved")

if __name__ == "__main__":
    main()
