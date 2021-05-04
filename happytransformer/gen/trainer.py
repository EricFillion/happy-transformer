"""

Fine-tuning for text generation models.

Based on the tutorial found here:
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
"""
from dataclasses import dataclass
import json
from transformers import default_data_collator
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from happytransformer.fine_tuning_util import preprocess_concatenate
from happytransformer.gen.default_args import ARGS_GEN_TRAIN, ARGS_GEN_EVAl
from datasets import load_dataset

@dataclass
class GENTrainArgs:
    learning_rate: float = ARGS_GEN_TRAIN["learning_rate"]
    num_train_epochs: int = ARGS_GEN_TRAIN["num_train_epochs"]
    batch_size: int = ARGS_GEN_TRAIN["batch_size"]
    weight_decay: float = ARGS_GEN_TRAIN["weight_decay"]
    adam_beta1: float = ARGS_GEN_TRAIN["adam_beta1"]
    adam_beta2: float = ARGS_GEN_TRAIN["adam_beta2"]
    adam_epsilon: float = ARGS_GEN_TRAIN["adam_epsilon"]
    max_grad_norm:  float = ARGS_GEN_TRAIN["max_grad_norm"]
    save_preprocessed_data: bool = ARGS_GEN_TRAIN["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_GEN_TRAIN["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_GEN_TRAIN["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_GEN_TRAIN["load_preprocessed_data_path"]
    preprocessing_processes: int = ARGS_GEN_TRAIN["preprocessing_processes"]
    mlm_probability: float = ARGS_GEN_TRAIN["mlm_probability"]


@dataclass
class GENEvalArgs:
    batch_size: int = ARGS_GEN_EVAl["batch_size"]
    save_preprocessed_data: bool = ARGS_GEN_EVAl["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_GEN_EVAl["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_GEN_EVAl["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_GEN_EVAl["load_preprocessed_data_path"]
    preprocessing_processes: int =ARGS_GEN_EVAl["preprocessing_processes"]
    mlm_probability: float = ARGS_GEN_EVAl["mlm_probability"]


class GENTrainer(HappyTrainer):
    """
    Trainer class for HappyWordPrediction
    """

    def train(self, input_filepath, dataclass_args: GENTrainArgs):
        """
        :param input_filepath: A file path to a text file that contains nothing but training data
        :param dataclass_args: A GENTrainArgs() object
        :return: None
        """

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
        """
        :param input_filepath: A file path to a text file that contains nothing but evaluating data
        :param dataclass_args: A GENEvalArgs() object
        :return: An EvalResult() object
        """

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

        result = self._run_eval(tokenized_dataset['eval'], default_data_collator, dataclass_args)

        return EvalResult(loss=result["eval_loss"])

    def test(self, input_filepath, solve, args):
        raise NotImplementedError()

    def _generate_json(self, json_path, dataset, name):
        """
        :param json_path: A path to a json file that will be created/overwritten
        :param dataset: A list of dictionaries that contain the keys "attention_mask," "input_ids" and "labels"
        :param name: A string to specify if the written data is for "Train" or "Eval"
        :return: None
        """
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
