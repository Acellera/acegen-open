import os
import logging
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import torch
from torch.utils.data import Dataset


def load_dataset(file_path):
    """Reads a list of SMILES from file_path"""

    smiles_list = []
    with open(file_path, "r") as f:
        for line in tqdm(f, desc="Load Samples"):
            smiles_list.append(line.split()[0])

    return smiles_list


class DeNovoDataset(Dataset):
    """Dataset that takes a list of smiles."""

    def __init__(self, cache_path, dataset_path, vocabulary, randomize_smiles=False):

        self.vocabulary = vocabulary
        self.dataset_path = dataset_path
        self.randomize_smiles = randomize_smiles
        os.makedirs(cache_path, exist_ok=True)

        self.files = {
            "smiles_index": "smiles.index.mmap",
            "smiles_data": "smiles.data.mmap",
        }
        self.files = {
            name: Path(cache_path, path).resolve() for name, path in self.files.items()
        }

        for path in self.files.values():
            if not path.exists():
                self._process()
                break

        self.mmaps = {
            name: np.memmap(path, mode="r", dtype=np.int64)
            for name, path in self.files.items()
        }

        if not (
                self.mmaps["smiles_index"][0] == 0
                and self.mmaps["smiles_index"][-1] == len(self.mmaps["smiles_data"])
        ):
            raise RuntimeError(
                "Error during the creation of Dataset memory maps. Incorrect indices detected."
            )

    def _sample_iter(self):

        smiles_list = load_dataset(self.dataset_path)

        for smile in tqdm(
                smiles_list,
                total=len(smiles_list),
                desc="Process samples",
        ):
            encoded_smile = self.vocabulary.encode_smile(smile)
            if encoded_smile is None:
                continue

            yield encoded_smile

    def _process(self):

        logging.info("Gathering statistics...")
        num_samples = 0
        num_smile_encodings = 0
        for encoded_smile in self._sample_iter():
            num_samples += 1
            num_smile_encodings += len(encoded_smile)

        mmaps = {}
        mmaps["smiles_index"] = np.memmap(
            str(self.files["smiles_index"]) + ".tmp",
            mode="w+",
            dtype=np.int64,
            shape=num_samples + 1,
            )
        mmaps["smiles_data"] = np.memmap(
            str(self.files["smiles_data"]) + ".tmp",
            mode="w+",
            dtype=np.int64,
            shape=num_smile_encodings,
            )

        logging.info("Storing data...")
        mmaps["smiles_index"][0] = 0
        for i, encoded_smile in enumerate(self._sample_iter()):
            i_begin = mmaps["smiles_index"][i]
            i_end = i_begin + len(encoded_smile)
            mmaps["smiles_data"][i_begin:i_end] = encoded_smile
            mmaps["smiles_index"][i + 1] = i_end

        for name, mmap in mmaps.items():
            mmap.flush()
            os.rename(mmap.filename, str(self.files[name]))

    def __getitem__(self, i):

        i_begin = self.mmaps["smiles_index"][i]
        i_end = self.mmaps["smiles_index"][i + 1]
        smile = torch.tensor(
            self.mmaps["smiles_data"][i_begin:i_end].copy(), dtype=torch.int64
        )

        if self.randomize_smiles:
            smile_string = self.vocabulary.remove_start_and_end_tokens(
                self.vocabulary.decode_smile(smile.tolist())
            )
            try:
                equivalent_sample = Chem.MolToSmiles(
                    Chem.MolFromSmiles(smile_string), doRandom=True, canonical=False
                )
                smile = torch.tensor(
                    self.vocabulary.encode_smile(equivalent_sample), dtype=torch.int64
                )
            except KeyError:  # Sometimes a token outside the vocabulary can appear
                pass

        return smile

    def __len__(self):
        return len(self.mmaps["smiles_index"]) - 1

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = torch.zeros(len(arr), max_length)
        for i, seq in enumerate(arr):
            collated_arr[i, : seq.size(0)] = seq
        return collated_arr
