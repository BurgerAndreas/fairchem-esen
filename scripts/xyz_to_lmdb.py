"""
Script to convert XYZ training data to LMDB files.

This script reads XYZ files matching the patterns from the config and converts
them to LMDB format for efficient training.

Usage example:
    # Convert training data
    python scripts/xyz_to_lmdb.py \\
        --data-path data/all/8020/train/ \\
        --output-path data/all/8020/train/ \\
        --pattern "*_train.xyz" \\
        --r-edges \\
        --radius 6.0 \\
        --max-neigh 30 \\
        --molecule-cell-size 20.0 \\
        --num-workers 4
    
    # Convert validation data
    python scripts/xyz_to_lmdb.py \\
        --data-path data/all/8020/val/ \\
        --output-path data/all/8020/val/ \\
        --pattern "*_validation.xyz" \\
        --r-edges \\
        --radius 6.0 \\
        --max-neigh 30 \\
        --molecule-cell-size 20.0 \\
        --num-workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
from pathlib import Path
from re import X

import ase.io
import lmdb
import numpy as np
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from tqdm import tqdm

from fairchem.core.datasets.atomic_data import AtomicData


def process_xyz_file(mp_arg):
    """Process a single XYZ file and write to LMDB database."""
    xyz_file, output_path, pid, args = mp_arg
    
    # Create LMDB filename: replace .xyz with .lmdb
    lmdb_filename = xyz_file.stem + ".lmdb"
    db_path = output_path / lmdb_filename
    
    db = lmdb.open(
        str(db_path),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    structures = ase.io.read(str(xyz_file), index=":")
    
    # Handle both single structure and list of structures
    if not isinstance(structures, list):
        structures = [structures]
    
    natoms_list = []
    idx = 0
    
    for i, atoms in enumerate(structures):
        # Extract energy from atoms.info
        # Try REF_energy first (as per config key_mapping), then energy
        energy_value = atoms.info.get("REF_energy", atoms.info.get("energy", None))
        
        # Copy REF_energy to energy in atoms.info so AtomicData.from_ase can use it
        # This ensures energy is stored in the data object
        if energy_value is not None:
            atoms.info["energy"] = energy_value
            # Also set in calculator for compatibility
            calc = SinglePointCalculator(atoms, energy=energy_value)
            atoms.calc = calc
        
        # Convert to AtomicData
        data_object = AtomicData.from_ase(
            atoms,
            r_energy=True,
            r_forces=False,
            r_edges=args.r_edges,
            radius=args.radius,
            max_neigh=args.max_neigh,
            molecule_cell_size=args.molecule_cell_size,
            sid=str(xyz_file),
        )
        
        # Store in LMDB
        txn = db.begin(write=True)
        txn.put(
            f"{idx}".encode("ascii"),
            pickle.dumps(data_object, protocol=-1),
        )
        txn.commit()
        
        natoms_list.append(len(atoms))
        idx += 1
    
    # Save count of objects in lmdb
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()
    
    db.sync()
    db.close()
    
    return natoms_list, xyz_file.name


def main(args: argparse.Namespace) -> None:
    """Main function to convert XYZ files to LMDB."""
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise RuntimeError(f"Data path does not exist: {data_path}")
    
    # Find XYZ files matching the pattern
    pattern = args.pattern
    xyz_files = sorted(data_path.glob(pattern))
    
    if not xyz_files:
        raise RuntimeError(f"No XYZ files found matching pattern '{pattern}' in {data_path}")
    
    print(f"Found {len(xyz_files)} XYZ files matching pattern '{pattern}': {xyz_files}")
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    num_workers = min(args.num_workers, len(xyz_files))
    
    # Process files in parallel
    mp_args = [
        (xyz_file, output_path, i % num_workers, args)
        for i, xyz_file in enumerate(xyz_files)
    ]
    
    all_natoms = []
    processed_files = []
    
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_xyz_file, mp_args)
        for natoms_list, filename in results:
            all_natoms.extend(natoms_list)
            processed_files.append(filename)
    
    # Create metadata.npz
    metadata_path = output_path / "metadata.npz"
    np.savez(metadata_path, natoms=np.array(all_natoms, dtype=np.int32))
    
    print(f"\nConversion complete!")
    print(f"Output directory: {output_path}")
    print(f"Total structures: {len(all_natoms)}")
    print(f"LMDB files created: {len(processed_files)}")
    print(f"Metadata file: {metadata_path}")


def get_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert XYZ training data to LMDB files"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to directory containing XYZ files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory to save LMDB files and metadata.npz",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.xyz",
        help="Glob pattern to match XYZ files (default: *.xyz)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--r-edges",
        action="store_true",
        help="Store edge indices in LMDB (default: False)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=6.0,
        help="Cutoff radius for graph construction (default: 6.0)",
    )
    parser.add_argument(
        "--max-neigh",
        type=int,
        default=30,
        help="Maximum number of neighbors (default: 30)",
    )
    parser.add_argument(
        "--molecule-cell-size",
        type=float,
        default=20.0,
        help="Cell size for molecules (default: 20.0)",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

