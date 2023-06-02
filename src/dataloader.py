from collections import OrderedDict

import pandas as pd
import torch


class DataLoader:
    def __init__(self, file_path, batch_size, nrows=0, skiprows=0):
        self.file_path = file_path
        self.batch_size = batch_size
        self.nrows = nrows - skiprows
        self.skipped = skiprows
        self._initialize_iterator(skiprows)

    def __len__(self):
        return self.nrows

    def _initialize_iterator(self, skiprows=0):
        self.iterator = pd.read_json(self.file_path, 
                                     lines=True, 
                                     chunksize=self.batch_size)
        if skiprows > 0:
            self.iterator.data.readlines(skiprows)
            self.iterator.nrows_seen = skiprows

    def _pad_targets(self, targets, padding_value=-1):
        # Find the length of the longest list
        max_len = max(len(sublist) for sublist in targets)
        # Pad each sublist to length m
        padded_targets = [
            sublist + [padding_value] * (max_len - len(sublist)) for sublist in targets
        ]
        return torch.LongTensor(padded_targets)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
            sentences = batch.context.to_list()
            targets = self._pad_targets(batch.targets)
            spans = [[tuple(s) for s in sp] for sp in batch.spans.to_list()]
            return sentences, spans, targets
        except StopIteration:
            self._initialize_iterator()  # Reset iterator
            raise StopIteration

    def get_state_dict(self):
        return OrderedDict(
            {
                "file_path": self.file_path,
                "batch_size": self.batch_size,
                "nrows": self.nrows,
                "rows_read": self.iterator.nrows_seen,
            }
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        dataloader = cls(
            state_dict["file_path"],
            state_dict["batch_size"],
            state_dict["nrows"],
            state_dict["rows_read"],
        )
        return dataloader
