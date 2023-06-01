from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import os
from typing import Union, cast, Sequence, List
from typing_extensions import Literal, get_args
from dotenv import load_dotenv
import numpy as np
from pathlib import Path
import copy
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig  # type: ignore
from huggingface_hub import snapshot_download
from accelerate import (
    init_empty_weights,
    dispatch_model,
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
from transformers import StoppingCriteria, StoppingCriteriaList
from eval_pipeline.dataset import (
    ClassificationExample,
    Example,
    ExampleWithClasses,
    LogoddsExample,
    SequenceProbExample,
    NumericExample,
    HitRateExample,
    TaskType,
)

from eval_pipeline.numeric_parser import BasicParser
from eval_pipeline.openai_api import APIParameters, BaseGPT3Model,InstructGPT3Model, OpenAIModel, call_api

from nltk.corpus import stopwords

from string import punctuation


OPENAI_API_BASE_URL = "https://api.openai.com/v1/engines"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# DEBUG: counting errors
error_count = 0
# for checking how long the input is
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

ValidHFModel = Literal[
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "gpt-neo-125M",
    "gpt-neo-1.3B",
    "gpt-neo-2.7B",
    "gpt-j-6B",
    "gpt-neox-20b",
    "opt-125m",
    "opt-350m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "T0pp",
    "T0_3B",
    "flan-t5-small",
    "flan-t5-base",
    "flan-t5-large",
    "flan-t5-xl",
    "flan-t5-xxl",
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b"
]
valid_hf_models: tuple[ValidHFModel, ...] = get_args(ValidHFModel)
# print(valid_hf_models)
# NOTE: due to limitations of get_args with nested Literals, we have to call it
# multiple times
valid_gpt3_models: tuple[OpenAIModel, ...] = get_args(OpenAIModel) # type: ignore
# print(OpenAIModel)
# print(valid_gpt3_models)
Device = Literal["cuda:0", "cpu"]


def clean_word(word):
    if len(word) < 2:
        return word
    
    word = word.lower()
    word = word.replace("�","")
    word = word.replace("’","")
    word = word.replace("”","")
    word = word.replace("“","")
    word = word.replace("‘","")
    word = word.replace('"','')
    word=word.strip('!')
    word=word.strip(';')
    word = word.strip('.')
    word = word.strip(',')
    word = word.strip(':')
    word = word.strip('?')
    word = word.strip()
    return word

def ignore_word_list():
    ignored_words = stopwords.words('english')
    return ignored_words

def ignore_token_list():
    punct = list(punctuation)
    blank_character = ['\r','\n', '\t', '~', '*',"”","�","’", "“","‘", '\n\n', '/']
    ret = list(set(punct + blank_character))
    return ret

def logprob_to_prob(logprob: float) -> float:
    return np.exp(logprob)


def top_hitrate(pred: List, pred_score: List, target: List, top_k: int = 5):
    """

    Args:
        pred: list of prediction tokens
        pred_score: list of predicted confidence score
        target: wrong prediction token list
        top_k: top-k for calculating hit rate

    Returns: result Dictionary

    """
    # print(pred)
    # print(pred_score)
    pred = [x.strip() for x in pred]
    pred_ = pred[:top_k]

    score = pred_score[:len(pred_)]
    # print(pred_)
    # print(score)
    hit_ = [1 if p in target else 0 for p in pred_]
    # print(hit_)
    hr = sum(hit_) / top_k
    # if score:
    w_hr = np.dot(np.array(hit_), np.array(score)) / sum(score)
    # else:
        # w_hr = 0.0
    return {"hr": hr, "w_hr": w_hr}


def get_whitespace_id(tokenizer):
    # stop_words_ids = [tokenizer.encode(stop_word) for stop_word in [" "]]
    # print("whitespace", stop_words_ids)
    # return stop_words_ids
    whitespace_id = int(tokenizer.convert_tokens_to_ids(" "))
    return whitespace_id

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = []):
      StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]



class Model(ABC):
    @abstractmethod
    def __call__(
        self, examples: list[Example], task_type: TaskType
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        raise NotImplementedError("Abstract method")

    @staticmethod
    def from_name(
        model_name: Union[ValidHFModel, OpenAIModel], device: Device
    ) -> Model:
        if model_name in valid_hf_models:
            model = HFModel(model_name, device)
        elif model_name in valid_gpt3_models:
            model = GPT3Model(model_name)
        else:
            raise ValueError(f"Unrecognised model '{model_name}'")
        return model


class HFModel(Model):
    def __init__(self, model_name: ValidHFModel, device: Device) -> None:
        self.device = device
        # have to download the opt models in advance since they're new
        # The OPT models have a start token that we need to remove in some places
        self.correction_for_start_token = 0
        self.model_name = model_name
        prefix = ''
        if model_name.startswith("opt-"):
            prefix = "facebook/"
            self.prefix = prefix
            self.model = self._load_opt(prefix + model_name, device)
            self.correction_for_start_token = 1
        elif model_name.startswith("gpt-neo") or model_name.startswith("gpt-j"):
            prefix = "EleutherAI/"
            torch.cuda.empty_cache()
            self.prefix = prefix
            self.model = AutoModelForCausalLM.from_pretrained(prefix + model_name, max_length=1024).to(self.device)  # type: ignore
        else:
            if 'flan' in model_name:
                prefix = 'google/'
                self.prefix = prefix
                self.model = AutoModelForSeq2SeqLM.from_pretrained(prefix + model_name, max_length=1024).to(self.device)  # type: ignore
            elif 't5' in model_name:
                prefix = ''
                self.prefix = prefix
                self.model = AutoModelForSeq2SeqLM.from_pretrained(prefix + model_name, max_length=1024).to(self.device)  # type: ignore
            elif 'T0' in model_name:
                prefix = 'bigscience/'
                self.prefix = prefix
                if 'pp' in model_name:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(prefix + model_name, revision="sharded", max_length=1024).to(self.device)  # type: ignore
                else:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(prefix + model_name, max_length=1024).to(self.device)  # type: ignore

            else:
                prefix = ''
                self.model = AutoModelForCausalLM.from_pretrained(prefix + model_name, max_length=1024).to(self.device)  # type: ignore

        # apparently the OPT models need slightly different tokenizers
        # https://huggingface.co/docs/transformers/main/en/model_doc/opt#overview
        if model_name.startswith("opt-"):
            use_fast = False
        else:
            use_fast = True
        # if model_name.startswith("gpt"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            prefix + model_name,
            model_max_length=1024,
            use_fast = use_fast,
            add_prefix_space = True
        )
        self.bad_word_ids = self.get_bad_words_ids()
    
    def stopping_criteria(self):
        stop_words_ids = get_whitespace_id(self.tokenizer)
        # print(stop_words_ids)
        stop_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_words_ids)])
        return stop_criteria


    def filter_word_list(self,word_list,scores, tokenizer, num):
        # ret_word_list = []
        ret_word_list = []

        ret_scores = []
        scores = torch.stack(list(scores), dim=0)
        for i, score in enumerate(scores):
            scores[i] = score.softmax(-1)    
        for i, word in enumerate(word_list):
            # print(i)
            # if  word not in ret_word_list:
            ret_word_list.append(word)
                # tokens = tokenizer(word, return_tensors="pt",add_special_tokens=False, truncation=True)
                # word_len = len(tokens['input_ids'][0])
            score = torch.max(scores[:, i, :],-1).values
                # print(score)
            ret_scores.append(score)
            if len(ret_word_list) == num:
                break
        ret_scores = torch.as_tensor(ret_scores)
        ret_scores = ret_scores.cpu().detach().numpy()
        # ret_dict = zip(ret_word_list, ret_scores)
        # ret_dict = sorted(ret_dict, key = lambda x:x[1], reverse= True)
        # print(ret_dict)
        # ret_word_list = [x[0] for x in ret_dict]
        # ret_scores = [x[1] for x in ret_dict]
        # print(ret_word_list)
        # print(ret_scores)
        # print(word_list)
        # print(ret_word_list, ret_scores)
        return ret_word_list, ret_scores


    def get_bad_words_ids(self):
        bad_words = ignore_word_list()
    
        ids_words = self.tokenizer(bad_words, add_special_tokens=False).input_ids
        # print(ids_words)
        bad_tokens = ignore_token_list()
        ids_tokens = self.tokenizer(bad_tokens, add_special_tokens = False).input_ids
        if 'gpt' in self.model_name:
            ignore_token_id = 220
        if [] in ids_tokens:
            ids_tokens.remove([])
        # print(ids_tokens)
        ids = []
        
        for i, id in enumerate(ids_words):
            if id not in ids:
                ids.append(id)
            # if ignore_token_id in id:
            #     id_ = copy.deepcopy(id)
            #     id_.remove(ignore_token_id)
            #     # print(id)
            #     if id_ != []:
            #         if id_ not in ids:
            #             ids.append(id_)

        for i, id in enumerate(ids_tokens):
            if id not in ids:
                ids.append(id)
            if 'gpt' in self.model_name:
                if ignore_token_id in id:
                    id_ = copy.deepcopy(id)
                    id_.remove(ignore_token_id)
                    # print(id)
                    if id_ != []:
                        if id_ not in ids:
                            ids.append(id_)
   
        if [] in ids:
            ids.remove([])
        # ids.append([50256])
        # ids.append([ignore_token_id])
        # print(ids)
        return ids

    # def get_bad_token_ids(self):
    #     bad_tokens = ignore_token_list()
    #     if 'gpt' in self.model_name:
    #         ids = self.tokenizer(bad_tokens, add_special_tokens=False).input_ids
    #     else:
    #         ids = self.tokenizer(bad_tokens, add_prefix_space=True, add_special_tokens=False).input_ids
    #     if [] in ids:
    #         ids.remove([])
        
    #     return ids

    def _load_opt(self, checkpoint: str, device: Device):
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
            max_length=1024,
        )
        return self.model

    def __call__(
        self, examples: list[Example], task_type: TaskType, write_dir, contrastive_search
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        # TODO: remove this restriction
        if len(examples) > 1:
            raise ValueError(
                f"Batch size of {len(examples)} not currently supported for HF models: please use 1"
            )
        with torch.no_grad():
            if task_type.startswith("classification"):
                classification_examples = cast("list[ClassificationExample]", examples)
                rv = self._evaluate_classification(
                    classification_examples, task_type=task_type
                )
                # print(rv)
            elif task_type == "numeric":
                numeric_examples = cast("list[NumericExample]", examples)
                rv = self._evaluate_numeric(numeric_examples)
            elif task_type == "sequence_prob":
                sequence_prob_examples = cast("list[SequenceProbExample]", examples)
                rv = self._evaluate_sequence_prob(sequence_prob_examples)
                # print(rv)
            elif task_type == "logodds":
                logodds_examples = cast("list[LogoddsExample]", examples)
                rv = self._evaluate_logodds(logodds_examples, take_absolute_value=False)
            elif task_type == "absolute_logodds":
                logodds_examples = cast("list[LogoddsExample]", examples)
                rv = self._evaluate_logodds(logodds_examples, take_absolute_value=True)
            elif task_type == "hitrate":
                hitrate_examples = cast("list[HitRateExample]", examples)
                rv = self._evaluate_hitrate(hitrate_examples, write_dir, contrastive_search)
                # print(rv)
            else:
                raise ValueError(f"Unrecognised task type {task_type}")
            return rv

    def _evaluate_classification(
        self,
        examples: list[ClassificationExample],
        task_type: TaskType,
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        prompts = [
            example.prompt + class_seq
            for example in examples
            for class_seq in example.classes
        ]
        all_logits, all_tokens = self._get_logits_and_tokens(prompts)
        total_logprobs = []
        losses = []
        labels_correct = []
        labels_predicted = []
        prompt_start = 0
        for example in examples:
            # print('example:', example)
            # print('prompt start:', prompt_start)
            n_classes = len(example.classes)
            # print('n_classes:', n_classes)
            class_logprobs = []
            for j in range(n_classes):
                class_index = prompt_start + j
                # print('class_index', class_index)
                class_logits = all_logits[class_index]
                # the lengths of each class sequence in tokens
                class_sequence = example.classes[j]
                # NOTE: we subtract 1 if OPT because the first token is the start of the sequence
                target_token_length = (
                    len(self.tokenizer(class_sequence)["input_ids"])
                    - self.correction_for_start_token
                )
                # print('target token length', target_token_length)
                # we only need the logits for the end sequence
                tokens = all_tokens[class_index]
                # print('tokens:', tokens)
                # we have to go back by one because we don't care about the logits for the predicted token
                sequence_logits = class_logits[-target_token_length - 1 : -1]
                sequence_tokens = tokens[-target_token_length:]
                # print('sequence_logits:', sequence_logits)
                # print('sequence_tokens:', sequence_tokens)
                # we take a log_softmax over all token logits for each position in the class sequence to
                #  get log probabilities, and then sum the logprobs for the tokens actually chosen
                logprobs = F.log_softmax(sequence_logits, dim=-1)
                class_logprob = sum(
                    [logprobs[i, token] for i, token in enumerate(sequence_tokens)]
                )
                class_logprobs.append(class_logprob.item())  # type: ignore (the sum is never empty so never just 0, always a tensor)

            total_logprob = torch.logsumexp(torch.tensor(class_logprobs), dim=-1).item()
            normalised_logprobs = F.log_softmax(torch.tensor(class_logprobs), dim=-1)
            loss = -normalised_logprobs[example.answer_index].item()
            label_correct = int(np.argmax(normalised_logprobs) == example.answer_index)
            total_logprobs.append(total_logprob)
            losses.append(loss)
            labels_correct.append(label_correct)

            label_predicted = example.classes[
                torch.tensor(class_logprobs).argmax(dim=-1).item()
            ]
            labels_predicted.append(label_predicted)

            prompt_start += n_classes
        return {
            "loss": losses,
            "correct": labels_correct,
            "predicted": labels_predicted,
            "total_logprob": total_logprobs,
        }

    def _get_logits_and_tokens(
        self, prompts: list[str]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        all_logits = []
        all_tokens = []
        for prompt in prompts:
            tokenized_inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True
            ).to(self.device)
            if 'T0' in self.model_name or 't5' in self.model_name:
                outputs = self.model(input_ids = tokenized_inputs.input_ids, decoder_input_ids = tokenized_inputs.input_ids)
            else:
                outputs = self.model(**tokenized_inputs)
            logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)
            # need to remove batch dimension
            all_logits.append(torch.squeeze(logits))
            all_tokens.append(torch.squeeze(tokenized_inputs["input_ids"]))
        return all_logits, all_tokens

    def _evaluate_sequence_prob(
        self, examples: list[SequenceProbExample]
    ) -> dict[str, Sequence[float]]:
        # finding the target
        prompts = [example.prompt + example.completion for example in examples]
        tokenized_inputs = self.tokenizer(
            prompts, return_tensors="pt", truncation=True
        ).to(self.device)

        target_sequences = [example.completion for example in examples]
        # NOTE: we have to apply the OPT token correction here too
        target_token_lengths = [
            len(self.tokenizer(word)["input_ids"]) - self.correction_for_start_token
            for word in target_sequences
        ]

        outputs = self.model(**tokenized_inputs)
        logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)

        losses = []
        for i in range(len(examples)):
            # we only need the logits for the end sequence
            tokens = tokenized_inputs["input_ids"][i]
            # we have to go back by one because we don't care about the logits for the predicted token
            sequence_logits = logits[i, -target_token_lengths[i] - 1 : -1]
            sequence_tokens = tokens[-target_token_lengths[i] :]
            logprobs = -F.log_softmax(sequence_logits, dim=-1)
            loss = sum([logprobs[i, token] for i, token in enumerate(sequence_tokens)])
            losses.append(loss.item())  # type: ignore (the sum is never empty so never just 0, always a tensor)
        return {"loss": losses}

    def _evaluate_hitrate(
        self, examples: list[HitRateExample], write_dir, contrastive_search
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        # finding the target
        prompts = [example.prompt for example in examples]
        # print(len(prompts))
        tokenized_inputs = self.tokenizer(
            prompts, return_tensors="pt", truncation=True
        ).to(self.device)

        # wrong_predictions = [example.completion for example in examples]

        
        # NOTE: we have to apply the OPT token correction here too
        # target_token_lengths = [
        #     len(self.tokenizer(word)["input_ids"]) - self.correction_for_start_token
        #     for word in wrong_predictions
        # ]
        
        num_beams = 5
        num_select_words = 5
        bad_words_ids = self.bad_word_ids
        # print(bad_words_ids)
        # bad_tokens_ids = self.get_bad_token_ids()
        # print('whitespace_ids',get_whitespace_id(self.tokenizer))
        # stopping_criteria = self.stopping_criteria
        # print(stopping_criteria)
        # whitespace_id = get_whitespace_id(self.tokenizer)
        # print(whitespace_id)
        max_new_tokens = 1
        if 't5' in self.model_name or 'T0' in self.model_name:
            if contrastive_search:
                generate_ids = self.model.generate(tokenized_inputs["input_ids"], max_length = max_new_tokens, bad_words_ids = bad_words_ids, num_beams = num_beams, num_return_sequences= num_beams, output_scores = True, return_dict_in_generate = True, penalty_alpha=0.6, top_k=10)
            else:
                generate_ids = self.model.generate(tokenized_inputs["input_ids"], max_length = max_new_tokens, bad_words_ids = bad_words_ids, num_beams = num_beams, num_return_sequences= num_beams, output_scores = True, return_dict_in_generate = True)

            generate_sequences = generate_ids.sequences
            _0_index = examples[0].prompt.index('<extra_id_0>')
            _result_prefix = examples[0].prompt[:_0_index]
            _result_suffix = examples[0].prompt[_0_index+12:]  # 12 is the length of <extra_id_0>
            end_token='<extra_id_1>'
        elif self.model_name.startswith('gpt'):
            if contrastive_search:
                generate_ids = self.model.generate(tokenized_inputs["input_ids"], max_new_tokens=max_new_tokens, bad_words_ids = bad_words_ids, num_beams = num_beams, num_return_sequences= num_beams, output_scores = True, return_dict_in_generate = True, pad_token_id = 50256, penalty_alpha=0.6, top_k=10)
            else:
                generate_ids = self.model.generate(tokenized_inputs["input_ids"], max_new_tokens=max_new_tokens, bad_words_ids = bad_words_ids, num_beams = num_beams, num_return_sequences= num_beams, output_scores = True, return_dict_in_generate = True, pad_token_id = 50256)

            generate_sequences = generate_ids.sequences[:, tokenized_inputs["input_ids"].shape[-1]:]
        
        else:
            if contrastive_search:
                generate_ids = self.model.generate(tokenized_inputs["input_ids"], max_new_tokens=max_new_tokens, bad_words_ids = bad_words_ids, num_beams = num_beams, num_return_sequences= num_beams, output_scores = True, return_dict_in_generate = True, penalty_alpha=0.6, top_k=10)
            else:
                generate_ids = self.model.generate(tokenized_inputs["input_ids"], max_new_tokens=max_new_tokens, bad_words_ids = bad_words_ids, num_beams = num_beams, num_return_sequences= num_beams, output_scores = True, return_dict_in_generate = True)
            generate_sequences = generate_ids.sequences[:, tokenized_inputs["input_ids"].shape[-1]:]
        # only keep the newly generated tokens
        res_1_list = []
        res_3_list = []
        res_5_list = []
        generated_strings = []
        generated_words = []
        # print(generate_ids.scores.shape)
        scores = list(generate_ids.scores)
        # print(generate_ids.sequences)
        for i in range(num_beams):
            generated_word = self.tokenizer.decode(generate_sequences[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # print(generated_word, '///', generate_sequences[i])
            if 't5' in self.model_name or 'T0' in self.model_name:
                if end_token in generated_word:
                    _end_token_index = generated_word.index(end_token)
                    generated_word = generated_word[:_end_token_index]
                # if end_token in _txt:
                #     _end_token_index = _txt.index(end_token)
                #     return _result_prefix + _txt[:_end_token_index] + _result_suffix
                # else:
                #     return _result_prefix + _txt + _result_suffix
            # if len(generated_word.split()) < 1:
                # print('generated_word',repr(generated_word), generate_sequences[i])
                # continue
            # generated_word = generated_word.split()[0]
            generated_word = clean_word(generated_word)
            # if generated_word == '':
            #     continue
            generated_words.append(generated_word)
        # select_words, select_scores = generated_words, scores
        # print(scores[0]) 
        select_words, select_scores = self.filter_word_list(generated_words, scores,self.tokenizer, num_select_words)
        # print(select_words)
        if len(select_words) < num_select_words:
            filler_words = ['<dummy>'] * (num_select_words-len(select_words))
            filler_scores = [0.0] * len(filler_words)
            select_words += filler_words
            select_scores += filler_scores
        for word in select_words:
            if 't5' in self.model_name or 'T0' in self.model_name:
                generated_string =  _result_prefix +  word + _result_suffix
            else:
                generated_string = examples[0].prompt + ' ' + word
            generated_strings.append(generated_string)    
        res_1 = top_hitrate(select_words, select_scores, examples[0].completion, top_k = 1)['w_hr']
        res_3 = top_hitrate(select_words, select_scores, examples[0].completion, top_k = 3)['w_hr']
        res_5 = top_hitrate(select_words, select_scores, examples[0].completion, top_k = 5)['w_hr']
        res_1_list.append(res_1)
        res_3_list.append(res_3)
        res_5_list.append(res_5)

        generated_file = Path(write_dir, self.model_name + '_generated.txt')
        with open(generated_file,'a') as f:
            for line in generated_strings:
                f.write(line + '\n')
        f.close()

        return {"WHR@1": res_1_list , 'WHR@3': res_3_list, 'WHR@5': res_5_list}

    def _evaluate_logodds(
        self,
        examples: list[LogoddsExample],
        take_absolute_value: bool = False,
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        """logodds is much like classification, except we need to compare across prompts so we just
        compute the log odds here"""
        prompts = [example.prompt for example in examples]
        other_prompts = [example.other_prompt for example in examples]
        tokenized_inputs = self.tokenizer(
            prompts, return_tensors="pt", truncation=True
        ).to(self.device)
        other_tokenized_inputs = self.tokenizer(
            other_prompts, return_tensors="pt", truncation=True
        ).to(self.device)
        outputs = self.model(**tokenized_inputs)
        other_outputs = self.model(**other_tokenized_inputs)
        # we only need the logits for the final (new) token
        # NOTE: this may need to change if we use batch size > 1 with padding
        logits = outputs["logits"][:, -1].detach().to(device="cpu", dtype=torch.float32)
        other_logits = (
            other_outputs["logits"][:, -1]
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        logodds = self._logodds_from_logits(examples, logits)
        other_logodds = self._logodds_from_logits(examples, other_logits)

        logodds_differences = list(np.array(logodds) - np.array(other_logodds))  # type: ignore (np typing bad)
        answer_indices = [example.answer_index for example in examples]
        # flip the order (and hence the sign) if the answer is "no"
        # (unless we are taking absolute values)
        for i, answer_index in enumerate(answer_indices):
            if answer_index == 1:
                logodds_differences[i] *= -1
            if take_absolute_value:
                logodds_differences[i] = np.abs(logodds_differences[i])

        accuracies = self._accuracies_from_logits(examples, other_logits)
        total_logprob = list(
            torch.logsumexp(
                torch.stack(
                    (
                        torch.tensor(
                            self._total_logprobs_from_logits(examples, logits)
                        ),
                        torch.tensor(
                            self._total_logprobs_from_logits(examples, other_logits)
                        ),
                    )
                ),
                dim=0,
            )
        )
        return {
            "logodds_difference": logodds_differences,
            "correct": accuracies,
            "total_logprob": total_logprob,  # type: ignore (they should be floats)
        }

    def _evaluate_numeric(
        self, examples: list[NumericExample]
    ) -> dict[str, Sequence[float]]:
        prompts = [example.prompt for example in examples]
        tokenized_inputs = self.tokenizer(
            prompts, return_tensors="pt", truncation=True
        ).to(self.device)
        parser = BasicParser()
        # NOTE: this may need to change if we use batch size > 1 with padding
        outputs = self.model.generate(
            **tokenized_inputs,
            do_sample=True,
            num_return_sequences=10,
            max_new_tokens=7,
            temperature=0.5,
            pad_token_id=50526,
        )
        full_completions = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        # strip out the prompt NOTE: again we're assuming the batch_size is 1
        untrimmed_completions = [
            fc[len(examples[0].prompt) :] for fc in full_completions
        ]
        # dropping anything after a new line
        completions = [comp.split("\n")[0] for comp in untrimmed_completions]
        floats = parser(completions)
        # for now, we'll just take the mean of valid outputs as the estimate
        valid_floats = [f for f in floats if f is not None]
        if len(valid_floats) > 0:
            estimate = sum(valid_floats) / len(valid_floats)
        else:
            raise ValueError("No valid numbers returned")
        return {"estimate": [estimate]}

    def _logodds_from_logits(
        self, examples: list[LogoddsExample], logits: torch.Tensor
    ) -> list[float]:
        """Given examples and logits for those examples,
        compute the binary log odds for each example"""
        logodds_list = []
        for i, example in enumerate(examples):
            relevant_logits = self._extract_relevant_logits(logits, example, i)
            logprobs = F.log_softmax(relevant_logits, dim=-1)
            # NOTE: assuming always binary
            if len(logprobs) != 2:
                raise ValueError(f"Expected len(logprobs) == 2, not {len(logprobs)}")
            logodds = logprobs[0] - logprobs[1]
            logodds_list.append(logodds.item())
        return logodds_list

    def _accuracies_from_logits(self, examples, logits) -> list[int]:
        """Given examples and logits for those examples,
        compute whether the predicted label is correct for each example"""
        labels_correct = []
        for i, example in enumerate(examples):
            relevant_logits = self._extract_relevant_logits(logits, example, i)
            label_correct = int(
                np.argmax(relevant_logits.cpu().detach().numpy())
                == example.answer_index
            )
            labels_correct.append(label_correct)
        return labels_correct

    def _extract_relevant_logits(
        self, logits: torch.Tensor, example: ExampleWithClasses, index: int
    ) -> torch.Tensor:
        example_logits = logits[index]
        # NOTE: we take the last element of the returned token list
        # this is because the tokenizer returns a 1-element list for GPT tokenizers
        # and a 2-element list with start token in the first position for OPT tokenizers
        class_tokens = [
            token[-1] for token in self.tokenizer(list(example.classes))["input_ids"]
        ]
        # log_softmax just subtracts a constant, so repeated applications change nothing
        # and there is no point in taking logprobs before focusing on the relevant indices
        relevant_logits = example_logits[class_tokens]
        return relevant_logits

    def _total_logprobs_from_logits(self, examples, logits) -> list[float]:
        """Given examples and logits for those examples,
        compute the classification loss for each example"""
        total_logprobs = []
        for i, example in enumerate(examples):
            example_logits = logits[i]
            # NOTE: we take the last element of the returned token list
            # this is because the tokenizer returns a 1-element list for GPT tokenizers
            # and a 2-element list with start token in the first position for OPT tokenizers
            class_tokens = [
                token[-1]
                for token in self.tokenizer(list(example.classes))["input_ids"]
            ]
            # log_softmax just subtracts a constant, so repeated applications change nothing
            # and there is no point in taking logprobs before focusing on the relevant indices
            example_logprobs = F.log_softmax(example_logits, dim=-1)
            relevant_logprobs = example_logprobs[class_tokens]
            total_logprobs.append(torch.logsumexp(relevant_logprobs, dim=-1).item())
        return total_logprobs


class GPT3Model(Model):
    def __init__(self, model_name: OpenAIModel) -> None:
        self.model_name: OpenAIModel = model_name

    def __call__(
        self, examples: list[Example], task_type: TaskType,  write_dir, contrastive_search
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:

        if task_type.startswith("classification"):
            classification_examples = cast("list[ClassificationExample]", examples)
            rv = self._evaluate_classification(classification_examples)
            # print(rv)
        elif task_type == "numeric":
            numeric_examples = cast("list[NumericExample]", examples)
            rv = self._evaluate_numeric(numeric_examples)
        elif task_type == "sequence_prob":
            SequenceProbExamples = cast("list[SequenceProbExample]", examples)
            rv = self._evaluate_sequence_prob(SequenceProbExamples)
            # print(rv)
        elif task_type == "logodds":
            logodds_examples = cast("list[LogoddsExample]", examples)
            rv = self._evaluate_logodds(logodds_examples, take_absolute_value=False)
        elif task_type == "absolute_logodds":
            logodds_examples = cast("list[LogoddsExample]", examples)
            rv = self._evaluate_logodds(logodds_examples, take_absolute_value=True)
        elif task_type == "hitrate":
            hitrate_examples = cast("list[HitRateExample]", examples)
            rv = self._evaluate_hitrate(hitrate_examples, write_dir, contrastive_search)
            # print(rv)
        else:
            raise ValueError(f"Unrecognised task type {task_type}")
        return rv


    def _evaluate_hitrate(
        self, examples: list[HitRateExample], write_dir, contrastive_search
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        full_prompts = [example.prompt for example in examples]
        # print(len(examples))
        
        # print(full_prompts)
        # return {"WHR@1": res_1_list , 'WHR@3': res_3_list, 'WHR@5': res_5_list}
        api_params = APIParameters(
            temperature=0.0,
            n=1,
            max_tokens=1,
            logprobs=5,
            stop=["\n"],
            echo=False,
        )
        # print('full_prompts', full_prompts)
        response_json = call_api(full_prompts, self.model_name, api_params).json()
        
        # print('response',response_json)
        # losses = []
        res_1_list = []
        res_3_list = []
        res_5_list = []
        generated_strings = []

        for i, example in enumerate(examples):
            tokens = []
            probs = []
            # print(example)
            text_index = len(example.prompt)
            # print(text_index)
            logprobs_dict = response_json["choices"][i]["logprobs"]
            # print('logprobs_dict',logprobs_dict)
            # print(pd.DataFrame(logprobs_dict))
            text_offset = logprobs_dict["text_offset"]
            actual_logprobs = logprobs_dict["top_logprobs"]
            try:
                token_index = text_offset.index(text_index)
            except ValueError as e:
                raise ValueError(
                    f"The target sequence '{example.completion}' did not start on a token boundary"
                )
            
            for key,value in actual_logprobs[0].items():
                word = clean_word(key)
                tokens.append(word)
                probs.append(logprob_to_prob(value))

            res_1 = top_hitrate(tokens, probs, example.completion, top_k=1)['w_hr']
            res_3 = top_hitrate(tokens, probs, example.completion, top_k=3)['w_hr']
            res_5 = top_hitrate(tokens, probs, example.completion, top_k=5)['w_hr']
            res_1_list.append(res_1)
            res_3_list.append(res_3)
            res_5_list.append(res_5)
            
            for word in tokens:
                generated_string = example.prompt + ' ' + word
                generated_strings.append(generated_string)
            
        generated_file = Path(write_dir, self.model_name + '_generated.txt')
        with open(generated_file,'a') as f:
            for line in generated_strings:
                f.write(line + '\n')
        f.close()

        return {"WHR@1": res_1_list , 'WHR@3': res_3_list, 'WHR@5': res_5_list}
        # return {"loss": losses}

    def _evaluate_classification(
        self,
        examples: list[ClassificationExample],
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        # making a prompt for each completion
        # NOTE: the effective batch size is now n times the parameter passed in (where n is number of classes)
        # but I'll fix that in the colab and it'll be fine
        prompts = [
            example.prompt + class_sequence
            for example in examples
            for class_sequence in example.classes
        ]

        api_params = APIParameters(
            temperature=0,
            n=1,
            max_tokens=0,
            logprobs=1,
            echo=True,
        )
        response_json = call_api(prompts, self.model_name, api_params).json()
        losses = []
        labels_correct = []
        labels_predicted = []
        total_logprobs = []
        choices = response_json["choices"]

        prompt_start = 0
        for example in examples:
            n_classes = len(example.classes)
            class_choices = choices[prompt_start : prompt_start + n_classes]

            # all class sequences begin after the initial prompt
            text_index = len(example.prompt)

            # accumulate logprobs for each class sequence separately
            relevant_logprobs = []
            for i in range(n_classes):
                logprobs_dict = class_choices[i]["logprobs"]
                text_offset = logprobs_dict["text_offset"]
                actual_logprobs = logprobs_dict["token_logprobs"]
                try:
                    token_index = text_offset.index(text_index)
                except ValueError as e:
                    raise ValueError(
                        f"The class sequence '{example.classes[i]}' did not start on a token boundary"
                    )
                class_logprob = 0
                for token_logprob in actual_logprobs[token_index:]:
                    class_logprob += token_logprob
                relevant_logprobs.append(class_logprob)

            relevant_logprobs = torch.tensor(relevant_logprobs)

            loss = -F.log_softmax(relevant_logprobs, dim=-1)[example.answer_index]
            losses.append(loss.item())
            total_logprob = torch.logsumexp(relevant_logprobs, dim=-1)
            total_logprobs.append(total_logprob.item())

            label_correct = int(np.argmax(relevant_logprobs) == example.answer_index)
            labels_correct.append(label_correct)

            label_predicted = example.classes[relevant_logprobs.argmax(dim=-1).item()]
            labels_predicted.append(label_predicted)

            prompt_start += n_classes
        return {
            "loss": losses,
            "correct": labels_correct,
            "predicted": labels_predicted,
            "total_logprob": total_logprobs,
        }

    def _evaluate_logodds(
        self,
        examples: list[LogoddsExample],
        take_absolute_value: bool = False,
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        prompts = [
            example.prompt + class_token
            for example in examples
            for class_token in example.classes
        ]
        other_prompts = [
            example.other_prompt + class_token
            for example in examples
            for class_token in example.classes
        ]
        api_params = APIParameters(
            temperature=0,
            n=1,
            max_tokens=0,
            logprobs=1,
            echo=True,
        )
        response_json = call_api(prompts, self.model_name, api_params).json()
        other_response_json = call_api(
            other_prompts, self.model_name, api_params
        ).json()
        logodds_differences = []
        labels_correct = []
        total_logprobs = []
        choices = response_json["choices"]
        other_choices = other_response_json["choices"]

        prompt_start = 0
        for example in examples:
            n_classes = len(examples[0].classes)
            class_choices = choices[prompt_start : prompt_start + n_classes]
            other_class_choices = other_choices[prompt_start : prompt_start + n_classes]

            relevant_logprobs = torch.tensor(
                [choice["logprobs"]["token_logprobs"][-1] for choice in class_choices]
            )
            other_relevant_logprobs = torch.tensor(
                [
                    choice["logprobs"]["token_logprobs"][-1]
                    for choice in other_class_choices
                ]
            )

            logodds = relevant_logprobs[0] - relevant_logprobs[1]
            other_logodds = other_relevant_logprobs[0] - other_relevant_logprobs[1]
            logodds_difference = logodds - other_logodds
            answer_index = example.answer_index
            # flip the order (and hence the sign) if the answer is "no"
            if answer_index == 1:
                logodds_difference *= -1

            if take_absolute_value:
                logodds_difference = np.abs(logodds_difference)

            logodds_differences.append(logodds_difference.item())
            total_logprob = (
                torch.logsumexp(
                    torch.cat((relevant_logprobs, other_relevant_logprobs)), dim=0
                ).item(),
            )
            total_logprobs.append(total_logprob)
            label_correct = int(
                np.argmax(other_relevant_logprobs) == example.answer_index
            )
            labels_correct.append(label_correct)

            prompt_start += n_classes
        return {
            "logodds_difference": logodds_differences,
            "correct": labels_correct,
            "total_logprob": total_logprobs,
        }

    def _evaluate_sequence_prob(
        self, examples: list[SequenceProbExample]
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        full_prompts = [example.prompt + example.completion for example in examples]
        api_params = APIParameters(
            temperature=0.0,
            n=1,
            max_tokens=0,
            logprobs=1,
            stop=["\n"],
            echo=True,
        )
        response_json = call_api(full_prompts, self.model_name, api_params).json()

        losses = []
        for i, example in enumerate(examples):
            text_index = len(example.prompt)
            logprobs_dict = response_json["choices"][i]["logprobs"]
            text_offset = logprobs_dict["text_offset"]
            actual_logprobs = logprobs_dict["token_logprobs"]
            try:
                token_index = text_offset.index(text_index)
            except ValueError as e:
                raise ValueError(
                    f"The target sequence '{example.completion}' did not start on a token boundary"
                )

            loss = 0
            for logprob in actual_logprobs[token_index:]:
                loss -= logprob
            losses.append(loss)

        return {"loss": losses}

    def _evaluate_numeric(
        self, examples: list[NumericExample]
    ) -> dict[str, Sequence[float]]:
        prompts = [example.prompt for example in examples]
        api_params = APIParameters(
            temperature=0.5,
            n=10,
            max_tokens=10,
            logprobs=None,
            stop=["\n"],
        )
        response_json = call_api(prompts, self.model_name, api_params).json()
        estimates = []
        choices = response_json["choices"]

        # working out which completions correspond to which input examples
        n_samples = len(choices) / len(examples)
        assert n_samples == int(n_samples)
        n_samples = int(n_samples)
        # parser = GPT3Parser("text-curie-001")
        parser = BasicParser()

        for i, example in enumerate(examples):
            start = i * n_samples
            completions = [
                choice["text"] for choice in choices[start : start + n_samples]
            ]
            floats = parser(completions)
            # for now, we'll just take the mean of valid outputs as the estimate
            valid_floats = [f for f in floats if f is not None]
            estimate = sum(valid_floats) / len(valid_floats)
            estimates.append(estimate)

        return {"estimate": estimates}
