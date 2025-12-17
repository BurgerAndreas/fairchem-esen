import os
import glob
import torch

chk_dirs = [
    "/scratch/aaldo/egsmole-data/test_dataset/amino_acids_angs",
    "/scratch/aaldo/egsmole-data/test_dataset/alcohols_angs",
    "/scratch/aaldo/egsmole-data/test_dataset/alkanes_angs",
    # "/scratch/aaldo/egsmole-data/test_dataset/pubchem_halfway",
    "/scratch/aaldo/egsmole-data/qm7_300_random_angs",
]


atomic_dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}

def read_data(data_path_single):
    atom_positions, atom_type = (
        torch.load(os.path.join(data_path_single, "atom_positions.pt")), 
        torch.load(os.path.join(data_path_single, "atom_type.pt")),
        )
    atoms = list(zip(atom_type, atom_positions))
    e_ccsd = torch.load(os.path.join(data_path_single, "e_corr_ccsd_fc.pt"))
    e_mp2 = torch.load(os.path.join(data_path_single, "e_corr_mp2_fc.pt"))

    # in case we are dealing with atomic numbers, convert to symbols
    symbols = [sym for sym, (x, y, z) in atoms]
    if all(sym in atomic_dict.keys() for sym in symbols):
        atoms = [(atomic_dict[sym], (x, y, z)) for sym, (x, y, z) in atoms]


    return atoms, e_ccsd - e_mp2

def save_xyz_file(data_dir_path, output_file):
    with open(output_file, "w") as out:
        for data_path_single in sorted(glob.glob(data_dir_path + "/*")):
            # print(f"Writing to {chk_path}")
            atoms, delta_e = read_data(data_path_single)
            if atoms is None:
                print(f"Skipping {data_path_single} due to read error")
                continue
            # print(f"{atoms}, delta_e={delta_e:.6f}")
            out.write(f"{len(atoms)}\nProperties=species:S:1:pos:R:3 REF_energy={delta_e}\n")
            for sym, (x, y, z) in atoms:
                out.write(f"{sym} {x:.8f} {y:.8f} {z:.8f}\n")

def main():
    for chk_dir in chk_dirs:
        data_dir_path = os.path.join(chk_dir, 'preprocessed', 'processed_data')
        
        output_file = os.path.basename(chk_dir)
        output_file = output_file[:-5] # remove _angs
        output_file += ".xyz" # add .xyz
        save_xyz_file(data_dir_path, output_file)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()