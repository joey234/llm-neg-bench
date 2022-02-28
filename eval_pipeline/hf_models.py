"""Code to handle the GPT-2 side of evaluation.
Uses the HuggingFace implementations of GPT-2.
Currently uses CPU because speed is not yet a concern.
"""
from __future__ import annotations
from typing_extensions import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
import torch
import torch.nn.functional as F
from pprint import pprint
from eval_pipeline.utils import YAxis
import logging


HFSize = Literal[
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "gpt-neo-125M",
    "gpt-neo-1.3B",
    "gpt-neo-2.7B",
    "gpt-j-6B",
]


class HFWrapper:
    def __init__(self, size: HFSize) -> None:
        # have to append the hoster if using Eleuther models
        prefix = ""
        if size.startswith("gpt-neo") or size.startswith("gpt-j"):
            prefix = "EleutherAI/"
        self.model = AutoModelForCausalLM.from_pretrained(prefix + size)
        self.tokenizer = AutoTokenizer.from_pretrained(prefix + size)
        self.id2token = {i: t for t, i in self.tokenizer.vocab.items()}

    def get_loss(
        self, text: str, answer_ix: int, possible_answers: tuple[str, str]
    ) -> float:
        """answer_ix gives the index of the intended answer from possible_answers"""
        logits = self.get_logits(text)
        # get the token id of the answer token
        positive_token_id, negative_token_id = self.tokenizer(possible_answers)[
            "input_ids"
        ]
        logprobs = F.log_softmax(logits, dim=-1)
        positive_logprob = logprobs[positive_token_id[0]]
        negative_logprob = logprobs[negative_token_id[0]]
        # DEBUG: checking alternative token choices
        other_pos, other_neg = self.tokenizer(
            [
                " 1",
                " 2",
            ]
        )["input_ids"]

        # For now I'm doing two log_softmaxes, which seems like it must be avoidable
        normalised_logprobs = F.log_softmax(
            torch.Tensor([positive_logprob, negative_logprob]), dim=-1
        )
        return -normalised_logprobs[answer_ix].item()

    def get_logits(self, text: str) -> torch.Tensor:
        encoded_input = self.tokenizer(text, return_tensors="pt")
        output = self.model(**encoded_input)
        raw_logits = output["logits"][0, -1]
        return raw_logits

    def get_logit_dict(self, text: str) -> dict[str, float]:
        logits = self.get_logits(text)
        logit_dict = {self.id2token[i]: logit for i, logit in enumerate(logits)}
        return logit_dict


def evaluate_hf_texts(
    text_answer_ix_pairs: list[tuple[str, int]],
    sizes: tuple[HFSize, ...],
    possible_answers: tuple[str, str],
) -> dict[str, dict[str, float]]:
    logging.info("CALLED HF")
    model_dict = {size: HFWrapper(size) for size in sizes}
    all_return_dicts = dict()

    for text, answer_ix in text_answer_ix_pairs:
        # for now, just using yes/no questions
        return_dict = dict()
        for size, model in model_dict.items():
            logging.info(f"RUNNING {size}")
            value = model.get_loss(text, answer_ix, possible_answers)
            return_dict[size] = value
        all_return_dicts[text] = return_dict
    return all_return_dicts