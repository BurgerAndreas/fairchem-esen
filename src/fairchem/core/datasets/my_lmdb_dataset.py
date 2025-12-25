import bisect
import pickle
from functools import cached_property
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from torch.utils.data import Dataset

from fairchem.core.common.registry import registry
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.modules.transforms import DataTransforms


GLOBAL_ATOM_NUMBERS = torch.tensor([1, 6, 7, 8, 9])
GLOBAL_ATOM_SYMBOLS = np.array(["H", "C", "N", "O", "F"])
Z_TO_ATOM_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}


def onehot_convert(atomic_numbers, device):
    """
    Convert a list of atomic numbers into an one-hot matrix
    """
    encoder = {
        1: [1, 0, 0, 0, 0],  # H
        6: [0, 1, 0, 0, 0],  # C
        7: [0, 0, 1, 0, 0],  # N
        8: [0, 0, 0, 1, 0],  # O
        9: [0, 0, 0, 0, 1],  # F
    }
    onehot = [encoder[i] for i in atomic_numbers]
    return torch.tensor(onehot, dtype=torch.int64, device=device)


@registry.register_dataset("lmdb")
class LmdbDataset(BaseDataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, config, transform=None, **kwargs):
        super().__init__(config)
        
        src = config["src"]
        self.path = Path(src)
        self.num_samples = 0
        self.transforms = DataTransforms(self.config.get("transforms", {}))
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                length = pickle.loads(
                    self.envs[-1].begin().get("length".encode("ascii"))
                )
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)
            try:
                # Try to get the stored length value first
                self.num_samples = pickle.loads(
                    self.env.begin().get("length".encode("ascii"))
                )
            except (TypeError, KeyError):
                # Fallback to entries count if length key doesn't exist
                self.num_samples = self.env.stat()["entries"]

            self._keys = [f"{j}".encode("ascii") for j in range(self.num_samples)]

        self.transform = transform

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.num_samples} samples"
            )

        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            if datapoint_pickled is None:
                raise KeyError(f"No data found for index {idx}")
            data_object = pickle.loads(datapoint_pickled)

        if self.transform is not None:
            data_object = self.transform(data_object)
        
        # # Handle one_hot encoding if present (convert to atomic numbers)
        # if hasattr(data_object, "one_hot") and not hasattr(data_object, "z"):
        #     indices = data_object.one_hot.long().argmax(dim=1)
        #     data_object.z = GLOBAL_ATOM_NUMBERS.to(data_object.pos.device)[
        #         indices.to(data_object.pos.device)
        #     ]

        # Apply transforms (this sets dataset attribute from config)
        data_object = self.transforms(data_object)

        data_object.dataset_idx = torch.tensor(idx)

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=1099511627776 * 2,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()


# SPLIT_CACHE = {}


# def _make_split_indices(total, split_fracs, permutation):
#     train_n = int(total * split_fracs[0])
#     val_n = int(total * split_fracs[1])
#     train_end = train_n
#     val_end = train_n + val_n
#     splits = {
#         "train": permutation[:train_end],
#         "val": permutation[train_end:val_end],
#         "test": permutation[val_end:],
#     }
#     return splits


# class AtomicLmdbDataset(Dataset):
#     def __init__(
#         self,
#         src,
#         split="train",
#         split_fracs=(0.8, 0.1, 0.1),
#         seed=0,
#         max_neigh=64,
#         cutoff=6.0,
#         molecule_cell_size=40.0,
#         dataset_name="sample",
#     ):
#         super().__init__()
#         self.base = LmdbDataset(src)
#         self.dataset_name = dataset_name
#         self.max_neigh = max_neigh
#         self.cutoff = cutoff
#         self.molecule_cell_size = molecule_cell_size
#         key = (str(Path(src).resolve()), seed)
#         if key not in SPLIT_CACHE:
#             rng = np.random.default_rng(seed)
#             SPLIT_CACHE[key] = rng.permutation(len(self.base))
#         splits = _make_split_indices(len(self.base), split_fracs, SPLIT_CACHE[key])
#         assert split in splits, f"split must be one of {list(splits)}"
#         self.indices = splits[split]
#         self.num_samples = len(self.indices)
#         self.split = split

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         base_idx = int(self.indices[idx])
#         base_data = self.base[base_idx]
#         atoms = Atoms(
#             numbers=base_data.z.cpu().numpy(),
#             positions=base_data.pos.cpu().numpy(),
#         )
#         calc_kwargs = {}
#         if hasattr(base_data, "energy"):
#             calc_kwargs["energy"] = float(base_data.energy)
#         if hasattr(base_data, "forces"):
#             calc_kwargs["forces"] = base_data.forces.cpu().numpy()
#         if calc_kwargs:
#             atoms.calc = SinglePointCalculator(atoms=atoms, **calc_kwargs)
#         atomic_data = AtomicData.from_ase(
#             atoms,
#             r_edges=True,
#             radius=self.cutoff,
#             max_neigh=self.max_neigh,
#             molecule_cell_size=self.molecule_cell_size,
#             task_name=self.dataset_name,
#             r_forces=True,
#         )
#         atomic_data.dataset_name = self.dataset_name
#         return atomic_data


if __name__ == "__main__":
    import os

    dataset_dir = os.path.expanduser(
        "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/"
    )
    dataset_files = [
        "ts1x-val.lmdb",
        "ts1x_hess_train_big.lmdb",
        "RGD1.lmdb",
    ]
    lmdb_path = os.path.join(dataset_dir, dataset_files[0])
    lmdb_dataset = LmdbDataset(lmdb_path)
    print("length of lmdb_dataset:", len(lmdb_dataset))
    print("first element of lmdb_dataset:", lmdb_dataset[0])
    print("first element of lmdb_dataset.pos:", lmdb_dataset[0].pos)
    # print("first element of lmdb_dataset.ae:", lmdb_dataset[0].ae)
    first_elem = lmdb_dataset[0]
    print("")
    print("hasattr(first_elem, 'hessian'):", hasattr(first_elem, "hessian"))
    print("'hessian' in first_elem:", "hessian" in first_elem)
