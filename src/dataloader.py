import ast

import pandas as pd
import torch


class DataLoader:
    def __init__(self, csv_path, batch_size):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self._initialize_iterator()

    def _initialize_iterator(self):
        self.iterator = pd.read_csv(
            self.csv_path,
            compression="infer",
            names=["sentences", "spans", "targets"],
            iterator=True,
            chunksize=self.batch_size,
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
            targets = self._pad_targets(batch.targets.map(ast.literal_eval))
            spans = batch.spans.map(ast.literal_eval).to_list()
            return sentences, spans, targets
        except StopIteration:
            self._initialize_iterator()  # Reset iterator
            raise StopIteration

    def reset(self):
        self._initialize_iterator()
