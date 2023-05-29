import torch
from transformers import AutoTokenizer


class LUKETokenizer:
    def __init__(self, model_id="studio-ousia/luke-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def decode_spans(self, outputs):
        mentions = []
        for idx, (positions, mask) in enumerate(
            zip(outputs.entity_position_ids, outputs.entity_attention_mask)
        ):
            mentions.append(
                [
                    self.tokenizer.decode(outputs.input_ids[idx][mention[mention > 0]])
                    for mention in positions[mask == 1].clamp(min=0)
                ]
            )
        return mentions

    def __call__(
        self,
        sentences,
        entity_spans,
        return_tensors="pt",
        padding=True,
        truncation=True,
        **kwargs
    ):
        inputs = self.tokenizer(
            sentences,
            entity_spans=entity_spans,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            **kwargs
        )

        del inputs['entity_ids']

        return inputs


class EnokeeTokenizer:
    def __init__(self, encoder: str, max_mention_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)
        self.max_mention_length = max_mention_length

    def _get_mention_pos_ids(
        self, offset_mapping: torch.Tensor, mention_spans: list[tuple[int, int]]
    ) -> torch.Tensor:
        # mask [#mentions, max_mention_length]
        mask = torch.full((len(mention_spans), self.max_mention_length), -1)
        for idx1, span in enumerate(mention_spans):
            idx2 = 0
            for pos, (s, e) in enumerate(offset_mapping):
                # if not pad token and span corresponding to
                # token in range of mention span
                if not (s == e == 0) and s >= span[0] and e <= span[1]:
                    mask[idx1, idx2] = pos
                    idx2 += 1
        return mask

    def _decode_mention(self, input_ids, pos_ids):
        mention_tokens = input_ids.index_select(0, pos_ids[pos_ids > 0])
        return self.tokenizer.decode(mention_tokens)

    def __call__(self, sentences, spans) -> dict:
        # encode inputs and create mask to get mention spans
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )

        # get positions of tokens corresponding to mentions
        inputs["mention_pos"] = []
        for i in range(inputs["offset_mapping"].shape[0]):
            mention_pos = self._get_mention_pos_ids(
                inputs["offset_mapping"][i], spans[i]
            )
            inputs["mention_pos"].append(mention_pos)
        inputs["mention_pos"] = torch.vstack(inputs["mention_pos"])

        # get positions to split mention position ids for each sentence
        inputs["mention_pos_splits"] = [len(i) for i in spans]

        # delete offsets as not required
        del inputs["offset_mapping"]

        return inputs
