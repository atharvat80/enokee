import pandas as pd
import ast

class DataLoader:
    def __init__(self, csv_path, batch_size):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.iterator = pd.read_csv(self.csv_path, 
                                    compression="gzip", 
                                    names=["sentences", "spans", "targets"],
                                    iterator=True, 
                                    chunksize=self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = self.iterator.get_chunk()
            sentences = batch.sentences.to_list()
            targets = batch.targets.map(ast.literal_eval)
            spans = batch.spans.map(ast.literal_eval).to_list()
            return sentences, spans, targets
        except StopIteration:
            raise StopIteration