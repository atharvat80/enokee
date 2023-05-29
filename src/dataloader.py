import ast
from collections import OrderedDict

import pandas as pd
import torch


class DataLoader:
    def __init__(self, csv_path, batch_size, skiprows=0, max_entities=32):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.max_entities = max_entities
        self._initialize_iterator(skiprows)

    def _initialize_iterator(self, skiprows):
        self.iterator = pd.read_csv(
            self.csv_path,
            compression="infer",
            names=["sentences", "spans", "targets"],
            iterator=True,
            chunksize=self.batch_size,
            skiprows=skiprows
        )

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
            batch = self.iterator.get_chunk()
            sentences = batch.sentences.to_list()
            # clip max targets
            targets = batch.targets.map(ast.literal_eval)
            targets = [i[:self.max_entities] for i in targets]
            targets = self._pad_targets(targets)
            # clip max spans
            spans = batch.spans.map(ast.literal_eval).to_list()
            spans = [i[:self.max_entities] for i in spans]
            return sentences, spans, targets
        except StopIteration:
            self._initialize_iterator()  # Reset iterator
            raise StopIteration

    def reset(self):
        self._initialize_iterator()

    def get_state_dict(self):
        return OrderedDict(
            {
                "csv_path": self.csv_path,
                "batch_size": self.batch_size,
                "rows_read": self.iterator._currow,
            }
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        dataloader = cls(state_dict["csv_path"], 
                         state_dict["batch_size"],
                         state_dict["rows_read"])
        return dataloader
