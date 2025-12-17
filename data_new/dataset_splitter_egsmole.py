import os
import torch
import json
import glob

atomic_dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}

train_file_name = 'qm7_300_random_train.xyz'
val_file_name = 'qm7_300_random_validation.xyz'

data_dir_path = '/scratch/aaldo/egsmole-data/qm7_300_random_angs/preprocessed/processed_data/'
splits_dir = '/project/aip-aspuru/aaldo/egsmole/splits/'

split_files = os.listdir(splits_dir) 
# 1090_splits_metadata.json  2080_splits_metadata.json  4060_splits_metadata.json  6040_splits_metadata.json  8020_splits_metadata.json
split_files = split_files[::-1] # reverse order to start from the largest training set


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

def load_all_data(data_dir_path):
    all_data = {}
    for data_path_single in sorted(glob.glob(data_dir_path + "/*")):
        all_data[os.path.basename(data_path_single)] = read_data(data_path_single)
    return all_data


def save_xyz_file(data_dict, output_file):
    with open(output_file, "w") as out:
        for file, (atoms, delta_e) in data_dict.items():
            # out.write(f"{len(atoms)}\nProperties=species:S:1:pos:R:3 REF_energy={delta_e} file={file}\n")
            out.write(f"{len(atoms)}\nProperties=species:S:1:pos:R:3 REF_energy={delta_e}\n")
            for sym, (x, y, z) in atoms:
                out.write(f"{sym} {x:.8f} {y:.8f} {z:.8f}\n")

def main():

    all_data = load_all_data(data_dir_path)

    for split in split_files:
        with open(os.path.join(splits_dir, split), 'r') as f:
            split_data = json.load(f)
        
        train_files = split_data['files']['train']
        val_files = split_data['files']['valid']

        train_data = {f: all_data[f] for f in train_files}
        val_data = {f: all_data[f] for f in val_files}

        dirname = split.split('_')[0]  # e.g., '1090'
        os.makedirs(dirname, exist_ok=True)

        train_file_name_path = f"{dirname}/{train_file_name}"
        val_file_name_path = f"{dirname}/{val_file_name}"

        save_xyz_file(train_data, train_file_name_path)
        save_xyz_file(val_data, val_file_name_path)


if __name__ == "__main__":
    main()


