import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from dscribe.descriptors import SOAP
from ase import Atoms
from ase.io import write
import sys
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
# Two-Stage MapReduce Diversity Selection:
# Stage 1: Process dataset in batches of BATCH_SIZE valid molecules.
#   For each batch: compute SOAP, run MaxMin to select 1%, store results.
# Stage 2: Combine all 1% samples, run MaxMin on combined SOAP features.
BATCH_SIZE = 50_000  # Process this many valid molecules per batch

SAMPLE_SIZES = [100, 1_000, 10_000]

# Local QCML TFDS location and read configuration
LOCAL_DATA_DIR = "/tmp"
READ_CONFIG = tfds.ReadConfig(interleave_cycle_length=1)

# SOAP Hyperparameters
# Optimized for organic small molecules
SPECIES = ["H", "C", "N", "O", "S", "F", "Cl", "P"] 
SPECIES_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "S": 16,
    "F": 9,
    "Cl": 17,
    "P": 15
}
RCUT = 6.0
N_MAX = 8
L_MAX = 6

# Allowed atomic numbers for SOAP descriptor (must match SPECIES)
ALLOWED_ATOMIC_NUMBERS = set(SPECIES_TO_Z.values())

# ==========================================
# 1. MapReduce Batch Processing
# ==========================================
def process_batches_mapreduce(batch_size):
    """
    Two-stage MapReduce approach: Stream through dataset, process in batches.
    For each batch:
    - Compute SOAP descriptors
    - Run MaxMin to select 1% of the batch
    - Store selected molecules, indices, and SOAP features
    Returns list of (Atoms, dataset_idx, SOAP_features) tuples for 1% samples.
    """
    print("Starting two-stage MapReduce batch processing...")
    
    # Setup Stream
    ds_z = tfds.load(
        "qcml/dft_atomic_numbers",
        split="full",
        as_supervised=False,
        data_dir=LOCAL_DATA_DIR,
        read_config=READ_CONFIG,
    )
    ds_r = tfds.load(
        "qcml/dft_positions",
        split="full",
        as_supervised=False,
        data_dir=LOCAL_DATA_DIR,
        read_config=READ_CONFIG,
    )
    
    num_entries = len(ds_z)
    print(f"Number of entries: {num_entries}")
    
    # Initialize SOAP descriptor (reused across batches)
    soap = SOAP(
        species=SPECIES,
        periodic=False,
        r_cut=RCUT,
        n_max=N_MAX,
        l_max=L_MAX,
        average="off"
    )
    
    # Accumulate 1% samples from all batches
    combined_1percent_samples = []
    
    # Current batch being built
    current_batch = []
    total_valid_seen = 0
    example_idx = 0
    batch_num = 0
    
    iterator = zip(ds_z, ds_r)
    
    for data_z, data_r in tqdm(iterator, total=num_entries):
        current_idx = example_idx
        z = data_z["atomic_numbers"].numpy()
        pos = data_r["positions"].numpy()
        
        # --- FILTERING ---
        # 1. Heavy Atoms <= 7
        if np.sum(z > 1) > 7:
            example_idx += 1
            continue
        
        # 2. Must contain F or Cl or S
        if not (9 in z or 17 in z or 16 in z):
            example_idx += 1
            continue
        
        # 3. All atomic numbers must be in allowed species (for SOAP compatibility)
        if not set(z).issubset(ALLOWED_ATOMIC_NUMBERS):
            example_idx += 1
            continue
        
        # If we pass filters, add to current batch
        mol = Atoms(numbers=z, positions=pos)
        current_batch.append((mol, current_idx))
        
        total_valid_seen += 1
        example_idx += 1
        
        # When batch is full, process it
        if len(current_batch) >= batch_size:
            batch_num += 1
            print(f"\nProcessing batch {batch_num} ({len(current_batch)} molecules)...")
            
            # Extract molecules and indices for this batch
            batch_mols = [mol for (mol, _) in current_batch]
            batch_indices = [idx for (_, idx) in current_batch]
            
            # Compute SOAP descriptors for this batch
            print(f"  Computing SOAP descriptors for batch {batch_num}...")
            features_list = soap.create(batch_mols, n_jobs=4)
            
            # Apply Max-Pooling
            pooled_features = []
            for f in features_list:
                pooled_features.append(np.max(f, axis=0))
            X_batch = np.array(pooled_features)
            
            # Downsample to 1% using MaxMin
            n_select = max(1, len(current_batch) // 100)
            print(f"  Running MaxMin to select {n_select} from {len(current_batch)} (1%)...")
            selected_indices = maxmin_sample(X_batch, n_select)
            
            # Store selected samples with their SOAP features
            for sel_idx in selected_indices:
                combined_1percent_samples.append((
                    batch_mols[sel_idx],
                    batch_indices[sel_idx],
                    X_batch[sel_idx]
                ))
            
            print(f"  Batch {batch_num} complete. Selected {len(selected_indices)} molecules.")
            print(f"  Total 1% samples so far: {len(combined_1percent_samples)}")
            
            # Clear batch to free memory
            current_batch = []
        
        # Periodic status update
        if total_valid_seen % 5000 == 0:
            sys.stdout.write(f"\rScanned valid: {total_valid_seen} | Current batch: {len(current_batch)} | Total 1% samples: {len(combined_1percent_samples)}")
            sys.stdout.flush()
    
    # Process remaining molecules in final partial batch
    if current_batch:
        batch_num += 1
        print(f"\nProcessing final batch {batch_num} ({len(current_batch)} molecules)...")
        
        batch_mols = [mol for (mol, _) in current_batch]
        batch_indices = [idx for (_, idx) in current_batch]
        
        print(f"  Computing SOAP descriptors for final batch...")
        features_list = soap.create(batch_mols, n_jobs=4)
        
        pooled_features = []
        for f in features_list:
            pooled_features.append(np.max(f, axis=0))
        X_batch = np.array(pooled_features)
        
        n_select = max(1, len(current_batch) // 100)
        print(f"  Running MaxMin to select {n_select} from {len(current_batch)} (1%)...")
        selected_indices = maxmin_sample(X_batch, n_select)
        
        for sel_idx in selected_indices:
            combined_1percent_samples.append((
                batch_mols[sel_idx],
                batch_indices[sel_idx],
                X_batch[sel_idx]
            ))
        
        print(f"  Final batch complete. Selected {len(selected_indices)} molecules.")
    
    print(f"\nFinished. Scanned {total_valid_seen} valid molecules.")
    print(f"Total 1% samples from all batches: {len(combined_1percent_samples)}")
    return combined_1percent_samples

# ==========================================
# 2. SOAP Feature Calculation (Max-Pooling)
# ==========================================
def compute_soaps(atoms_list):
    print("\nComputing SOAP descriptors...")
    
    # Initialize SOAP
    # average="off" is CRITICAL for Max-Pooling
    soap = SOAP(
        species=SPECIES,
        periodic=False,
        r_cut=RCUT,
        n_max=N_MAX,
        l_max=L_MAX,
        average="off" 
        # r_cut=None,
        # n_max=None,
        # l_max=None,
        # sigma=1.0,
        # rbf="gto",
        # weighting=None,
        # average="off",
        # compression={"mode": "off", "species_weighting": None},
        # species=None,
        # periodic=False,
        # sparse=False,
        # dtype="float64",
    )
    
    # Calculate features
    # This returns a list of arrays (one array per molecule: [n_atoms, n_features])
    features_list = soap.create(atoms_list, n_jobs=4)
    
    # Apply Max-Pooling
    # We take the max value for each feature across all atoms in the molecule
    pooled_features = []
    for f in features_list:
        # f.shape is (n_atoms, n_features)
        # max(axis=0) collapses it to (n_features,)
        pooled_features.append(np.max(f, axis=0))
        
    return np.array(pooled_features)

# ==========================================
# 3. MaxMin (Farthest Point) Sampling
# ==========================================
def maxmin_sample(X, n_samples):
    """
    Greedy Farthest Point Sampling.
    Returns indices of the selected samples.
    """
    n_total = X.shape[0]
    if n_samples > n_total:
        print(f"Warning: Requested {n_samples} but only have {n_total}. Returning all.")
        return list(range(n_total))

    # Start with a random seed
    selected_indices = [np.random.randint(0, n_total)]
    
    # Initialize distances with respect to the first point
    # We maintain the "minimum distance to the set" for every point
    min_dists = np.linalg.norm(X - X[selected_indices[0]], axis=1)
    
    print(f"Sampling {n_samples} geometries...")
    
    for i in range(n_samples - 1):
        # Pick the point that has the LARGEST minimum distance to the current set
        farthest_point_idx = np.argmax(min_dists)
        selected_indices.append(farthest_point_idx)
        
        # Update distances: compute dist to NEW point, keep the minimum
        new_dists = np.linalg.norm(X - X[farthest_point_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        
        if i % 100 == 0:
             sys.stdout.write(f"\rSelected {len(selected_indices)}/{n_samples}")
             sys.stdout.flush()
             
    print(f"\rSelected {n_samples}/{n_samples}          ")
    return selected_indices

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Stage 1 (Map): Process all batches, compute SOAP per batch, downsample to 1%
    combined_1percent = process_batches_mapreduce(BATCH_SIZE)
    if not combined_1percent:
        print("No matching molecules found.")
        sys.exit()

    print(f"\nCombined 1% samples from all batches: {len(combined_1percent)} molecules.")

    # Split into molecules, indices, and SOAP features
    pool_atoms = [mol for (mol, _, _) in combined_1percent]
    pool_indices = [idx for (_, idx, _) in combined_1percent]
    # Extract SOAP features (already computed during batch processing)
    X_features = np.array([features for (_, _, features) in combined_1percent])
    print(f"Combined SOAP feature matrix shape: {X_features.shape}")

    # 2. Stage 2 (Reduce): Run MaxMin sampling on combined 1% samples
    # Since MaxMin is ordered (1st is farthest from 0, 2nd is farthest from {0,1}...), 
    # taking the first N items of a MaxMin run is effectively a valid MaxMin set of size N.
    max_k = max(SAMPLE_SIZES)
    print(f"\nRunning final MaxMin sampling on {len(X_features)} combined 1% samples...")
    indices_sorted = maxmin_sample(X_features, max_k)
    
    # 3. Save Outputs
    for k in SAMPLE_SIZES:
        if k > len(indices_sorted):
            continue
            
        # Positions into the combined 1% pool
        subset_positions = indices_sorted[:k]
        subset_mols = [pool_atoms[i] for i in subset_positions]
        subset_dataset_indices = [pool_indices[i] for i in subset_positions]

        # Embed dataset indices as per-structure XYZ comments
        comments = [f"dataset_idx={idx}" for idx in subset_dataset_indices]
        
        filename = f"diverse_qcml_{k}.xyz"
        write(filename, subset_mols, comment=comments)
        print(f"Saved {k} diverse geometries to {filename}")

    # Example: reading dataset indices back from an XYZ file with ASE
    # from ase.io import read
    # images = read("diverse_qcml_100.xyz", index=":")
    # comments = [img.info.get("comment", "") for img in images]

