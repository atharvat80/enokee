import bz2
import json
from collections import OrderedDict

import torch


class DataLoader:
    def __init__(self, file_path, batch_size, n_rows=0, skip_lines=0):
        self.file_path = file_path
        self.batch_size = batch_size
        self.n_rows = n_rows
        self.lines_read = skip_lines
        self._initialize_reader(skip_lines)
        self.end_of_file = False

    def _initialize_reader(self, skip_lines):
        self.reader = bz2.open(self.file_path, "rt")
        for _ in range(skip_lines):
            next(self.reader)

    def _pad_targets(self, targets, padding_value=-1):
        # Find the length of the longest list
        max_len = max(len(sublist) for sublist in targets)
        # Pad each sublist to length m
        padded_targets = [
            sublist + [padding_value] * (max_len - len(sublist)) for sublist in targets
        ]
        return torch.LongTensor(padded_targets)

    def __len__(self):
        return self.n_rows - self.lines_read

    def __del__(self):
        self.reader.close()

    def __iter__(self):
        return self

    def __next__(self):
        contexts, spans, targets = [], [], []
        for _ in range(self.batch_size):
            line = self.reader.readline()
            # check for end of file
            if not line:
                self.end_of_file = True
                break
            # else parse entry
            self.lines_read += 1
            entry = json.loads(line)
            contexts.append(entry["context"])
            targets.append(entry["targets"])
            spans.append([tuple(i) for i in entry["spans"]])

        if not contexts:
            self.reader.seek(0)
            self.lines_read = 0
            self.end_of_file = False
            raise StopIteration

        return contexts, spans, self._pad_targets(targets)

    def get_state_dict(self):
        return OrderedDict(
            {
                "file_path": self.file_path,
                "batch_size": self.batch_size,
                "n_rows": self.n_rows,
                "lines_read": self.lines_read,
            }
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        dataloader = cls(
            state_dict["file_path"],
            state_dict["batch_size"],
            n_rows=state_dict["n_rows"],
            skip_lines=state_dict["lines_read"],
        )
        return dataloader
