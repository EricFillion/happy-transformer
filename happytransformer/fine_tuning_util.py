"""
Based on
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
and
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
"""

from dataclasses import dataclass
from typing import Union

from datasets import Dataset
from transformers import PreTrainedTokenizer, TrainerCallback

from happytransformer.args import GENTrainArgs, WPTrainArgs, GENEvalArgs, WPEvalArgs

# Used for text gen and mlm fine-tuning.
def tok_text_gen_mlm(tokenizer: PreTrainedTokenizer, dataset: Dataset, args: Union[GENTrainArgs, WPTrainArgs, GENEvalArgs, WPEvalArgs],  mlm: bool=True) -> Dataset:
    #todo set to args.max_length
    if args.max_length is None:
        max_input_length = tokenizer.model_max_length
    else:
        max_input_length= args.max_length

    def tokenize_function(example):
        texts = example["text"]
        # Add newlines back that were removed from load_dataset().
        texts = [case + "\n" for case in texts[:]]
        return tokenizer(texts)

    tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                      remove_columns=["text"])

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        output_length = (total_length // max_input_length) * max_input_length

        # if  total_length is less than the max_input_length length
        # then it causes an error unless the code below is used.
        # this is due to total_length being truncated to 0
        if output_length == 0:
            output_length = total_length

        result = {
            k: [t[i: i + max_input_length] for i in range(0, output_length, max_input_length)]
            for k, t in concatenated_examples.items()
        }

        if not mlm:
            # Masked language models don't need labels. Text generation models do
            result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True)


    return tokenized_dataset


def csv_tok_text_gen_mlm(tokenizer: PreTrainedTokenizer, dataset: Dataset, args: Union[GENTrainArgs, WPTrainArgs, GENEvalArgs, WPEvalArgs],  mlm=True) -> Dataset:
    if args.max_length is None:
        max_input_length = tokenizer.model_max_length
    else:
        max_input_length = args.max_length

    def tokenize_function(example):
        texts = example["text"]
        toks = tokenizer(texts, padding="max_length", truncation=True, max_length=max_input_length)
        if not mlm:
            toks["labels"] = toks["input_ids"]
        return toks

    dataset= dataset.map(tokenize_function,
                batched=True,
                remove_columns=["text"])

    return dataset

@dataclass
class EvalResult:
    loss: float


ZERO_2_SETTINGS = {
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
    },

    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "last_batch_iteration": -1,
            "total_num_steps": "auto",
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 32,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}

ZERO_3_SETTINGS = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "last_batch_iteration": -1,
            "total_num_steps": "auto",
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 8,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}


class FistStep(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_log = True
            control.should_evaluate = True

