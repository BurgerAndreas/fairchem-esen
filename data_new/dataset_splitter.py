import os
import random

random.seed(42)

file_name = 'qm7_300_random.xyz'

train_file_name = 'qm7_300_random_train.xyz'
val_file_name = 'qm7_300_random_validation.xyz'

splits_train = [0.8, 0.6, 0.4, 0.2, 0.1]
splits_val = [1 - split for split in splits_train]

def read_molecules(file_name):
    molecules_list = []
    with open(file_name, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            num_atoms = int(line)
            mol_lines = lines[i:i + num_atoms + 2]
            mol_block = ''.join(mol_lines).rstrip('\n')
            molecules_list.append(mol_block)
            i += num_atoms + 2
        else:
            i += 1

    return molecules_list

molecules_list = read_molecules(file_name)
# print(f"Total molecules read: {len(molecules_list)}")
# print(molecules_list[0])


for split_train in splits_train:
    num_molecules = len(molecules_list)
    num_train = int(num_molecules * split_train)
    num_val = num_molecules - num_train

    indices = list(range(len(molecules_list)))
    random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    train_molecules = [molecules_list[i] for i in train_indices]
    val_molecules = [molecules_list[i] for i in val_indices]

    dirname = f"{int(split_train*100)}{int((100-split_train*100))}"
    os.makedirs(dirname, exist_ok=True)

    train_file = f"{dirname}/{train_file_name}"
    val_file = f"{dirname}/{val_file_name}"

    with open(train_file, "w") as f:
        for mol in train_molecules:
            f.write(mol + "\n")

    with open(val_file, "w") as f:
        for mol in val_molecules:
            f.write(mol + "\n")

    print(f"Created {train_file} with {len(train_molecules)} molecules")
    print(f"Created {val_file} with {len(val_molecules)} molecules")
    print("-----")

