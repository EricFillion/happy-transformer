"""

Fine-tuning for text generation odels.

Based on the tutorial found here:
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
"""
from dataclasses import dataclass
import json
from transformers import default_data_collator
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from happytransformer.fine_tuning_util import preprocess_concatenate
from datasets import load_dataset


@dataclass
class GENTrainArgs:
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float
    num_train_epochs: int
    preprocessing_processes: int
    mlm_probability: float

    save_preprocessed_data: False
    save_preprocessed_data_path: ""
    load_preprocessed_data: False
    load_preprocessed_data_path: ""


@dataclass
class GENEvalArgs:
    preprocessing_processes: int
    mlm_probability: float

    save_preprocessed_data: False
    save_preprocessed_data_path: ""
    load_preprocessed_data: False
    load_preprocessed_data_path: ""

class GENTrainer(HappyTrainer):
    """
    Trainer class for HappyWordPrediction
    """

    def train(self, input_filepath, dataclass_args: GENTrainArgs):

        if not dataclass_args.load_preprocessed_data:
            self.logger.info("Preprocessing dataset...")
            dataset = load_dataset("text", data_files={"train": input_filepath})
            tokenized_dataset = preprocess_concatenate(self.tokenizer, dataset, dataclass_args.preprocessing_processes, False)

        else:
            self.logger.info("Loading dataset from %s...", dataclass_args.load_preprocessed_data_path)
            tokenized_dataset = load_dataset("json", data_files={"train": dataclass_args.load_preprocessed_data_path}, field='train')

        if dataclass_args.save_preprocessed_data:
            if dataclass_args.load_preprocessed_data:
                self.logger.warning("Both save_preprocessed_data and load_data are enabled,")

            self.logger.info("Saving training dataset to %s...", dataclass_args.save_preprocessed_data_path)

            self._generate_json(dataclass_args.save_preprocessed_data_path, tokenized_dataset["train"], "train")

        self.logger.info("Training...")

        self._run_train(tokenized_dataset['train'], dataclass_args, default_data_collator)

    def eval(self, input_filepath, dataclass_args: GENEvalArgs):

        if not dataclass_args.load_preprocessed_data:
            self.logger.info("Preprocessing dataset...")
            datasets = load_dataset("text", data_files={"eval": input_filepath})
            tokenized_dataset = preprocess_concatenate(self.tokenizer, datasets, dataclass_args.preprocessing_processes, False)

        else:
            self.logger.info("Loading dataset from %s...", dataclass_args.load_preprocessed_data_path)
            tokenized_dataset = load_dataset("json", data_files={"eval": dataclass_args.load_preprocessed_data_path}, field='eval')

        if dataclass_args.save_preprocessed_data:
            if dataclass_args. load_preprocessed_data:
                self.logger.warning("Both save_preprocessed_data and load_data are enabled.")

            self.logger.info("Saving evaluating dataset to %s...", dataclass_args.save_preprocessed_data_path)

            self._generate_json(dataclass_args.save_preprocessed_data_path, tokenized_dataset["eval"], "eval")

        self.logger.info("Evaluating...")

        result = self._run_eval(tokenized_dataset['eval'], default_data_collator)

        return EvalResult(loss=result["eval_loss"])

    def test(self, input_filepath, solve, args):
        raise NotImplementedError()

    def _generate_json(self, json_path, dataset, name):
        data = {}
        data[name] = []
        data = {
            name: [
                {
                    'attention_mask': case['attention_mask'],
                    'input_ids': case['input_ids'],
                    'labels': case['labels']
                }
                for case in dataset
            ]
        }

        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)
