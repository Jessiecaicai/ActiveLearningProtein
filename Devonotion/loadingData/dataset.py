from pathlib import Path

import lmdb
import pickle as pkl
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from torch.utils.data import Dataset
import numpy as np
import torch
from .tokenizers import Tokenizer


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

class LMDBDataset(Dataset):

    def __init__(self,
                 data_file):

        # data_path = "/research/zqg/dataset"
        data_path = "/home/guo/data/tape_data"
        # data_path = "/root/pyProject"
        # data_file = data_file
        data_location = Path(data_path + "/" + data_file)

        if not data_location.exists():
            raise FileNotFoundError(data_location)

        env = lmdb.open(str(data_location), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))  # 数据集总数量

        self._env = env
        self._num_examples = num_examples

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        with self._env.begin(write=False) as txn:
            item = pkl.loads(txn.get(str(index).encode()))
            if 'id' not in item:
                item['id'] = str(index)

        return item

# task('fluorescence')
# 前五百条序列训练
class FluorescenceDatasetFiveHundreds(Dataset):

    def __init__(self,
                 split: str,
                 tokenizer: Union[str, Tokenizer] = "iupac"):

        data_file = f"fluorescence/fluorescence_{split}.lmdb"
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.data = LMDBDataset(data_file)

        # list_five_hundreds = []
        # for each_item in self.data_all:
        #     if each_item['id'] in range(500):
        #         list_five_hundreds.append(each_item)

        # if self.data_all['id'] ==
        # self.data = list_five_hundreds

    def __len__(self) -> int:
        # return len(self.data)
        return 500

    def __getitem__(self, index: int):
        item = self.data[index]
        seq_length = item['protein_length']
        if seq_length <= 256:
            item['primary'] = item['primary']
        else:
            item['primary'] = item['primary'][:256]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, float(item['log_fluorescence'][0])

    def collaten_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, fluorescence_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 1))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 1))
        fluorescence_true_value = torch.FloatTensor(fluorescence_true_value)  # type: ignore
        fluorescence_true_value = fluorescence_true_value.unsqueeze(1)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': fluorescence_true_value}

