"""
Based on
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
and
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
"""

from dataclasses import dataclass

from datasets import Dataset
from transformers import PreTrainedTokenizer

# Used for text gen and mlm fine-tuning.
#todo rename
def tok_text_gen_mlm(tokenizer: PreTrainedTokenizer, dataset: Dataset, preprocessing_processes: int =1, mlm=True):

    max_input_length = tokenizer.model_max_length

    def tokenize_function(example):
        texts = example["text"]
        # Add newlines back that were removed from load_dataset().
        texts = [case + "\n" for case in texts[:]]
        return tokenizer(texts)

    tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                      num_proc=preprocessing_processes,
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
        batched=True,
        num_proc=preprocessing_processes,
    )


    return tokenized_dataset


@dataclass
class EvalResult:
    loss: float