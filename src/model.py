from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.utils import logging

logging.set_verbosity_error()


class EnokeeConfig:
    def __init__(
        self,
        d_model=768,
        n_heads=12,
        d_ff=1028,
        n_layers=4,
        dropout=0.1,
        num_entities=51000,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.num_entities = num_entities
        self.dropout = dropout

    def to_dict(self):
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "num_entities": self.num_entities,
            "dropout": self.dropout,
        }


class EnokeeEncoder(nn.Module):
    """SPOT style encoder"""

    def __init__(self, config, base_model_id="roberta-base"):
        super().__init__()
        self.config = config
        # RoBERTa backend
        self.base_model = AutoModel.from_pretrained(base_model_id)
        for param in self.base_model.parameters():
            param.requires_grad = False
        # Attention for mention tokens
        self.attention = torch.nn.Linear(config.d_model, 1)
        # Encoder
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_ff,
                    batch_first=True,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 100, bias=False), 
            nn.Linear(100, config.num_entities)
        )

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def get_state_dict(self):
        state_dict = OrderedDict(
            {
                key: params
                for key, params in self.state_dict().items()
                if not (key.startswith("base_model"))
            }
        )
        return state_dict

    def forward(self, input_ids, entity_position_ids, entity_attention_mask, **kwargs):
        # encode context
        last_hidden_state = self.base_model(input_ids)["last_hidden_state"]
        # gather mention tokens
        batch_size, max_mentions, max_mention_len = entity_position_ids.shape
        mentions = torch.zeros(
            batch_size,
            max_mentions,
            max_mention_len,
            self.d_model,
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype
        )
        for i in range(batch_size):
            for j in range(max_mentions):
                if entity_attention_mask[i, j] == 0:
                    continue
                for x in range(max_mention_len):
                    if entity_position_ids[i, j, x] == -1:
                        break
                    mentions[i, j, x, :] = last_hidden_state[i, x, :]

        # apply self attention to combine tokens of a mention into a singel vector
        scores = nn.functional.softmax(self.attention(mentions).squeeze(-1), dim=-1)
        inputs = torch.matmul(scores.unsqueeze(-2), mentions).squeeze(-2)

        # pass the mention embeddings through encoder
        for layer in self.layers:
            inputs = layer(inputs)

        return self.classifier(inputs)
