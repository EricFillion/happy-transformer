"""
Fine-tuning for masked word prediction models.

Based on the tutorial found here:
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
"""
from dataclasses import dataclass
from transformers import DataCollatorForLanguageModeling
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from datasets import load_dataset
from happytransformer.fine_tuning_util import preprocess_concatenate
from happytransformer.wp.default_args import ARGS_WP_TRAIN, ARGS_WP_EVAl, ARGS_WP_TEST
from happytransformer.happy_trainer import TrainArgs
from typing import Union

import json


@dataclass
class WPTrainArgs(TrainArgs):
    save_preprocessed_data: bool = ARGS_WP_TRAIN["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_WP_TRAIN["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_WP_TRAIN["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_WP_TRAIN["load_preprocessed_data_path"]
    preprocessing_processes: int = ARGS_WP_TRAIN["preprocessing_processes"]
    mlm_probability: float = ARGS_WP_TRAIN["mlm_probability"]
    line_by_line: bool = ARGS_WP_TRAIN["line_by_line"]


@dataclass
class WPEvalArgs:
    batch_size: int = ARGS_WP_EVAl["batch_size"]
    save_preprocessed_data: bool = ARGS_WP_EVAl["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_WP_EVAl["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_WP_EVAl["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_WP_EVAl["load_preprocessed_data_path"]
    preprocessing_processes: int =ARGS_WP_EVAl["preprocessing_processes"]
    mlm_probability: float = ARGS_WP_EVAl["mlm_probability"]
    line_by_line: bool = ARGS_WP_EVAl["line_by_line"]



class WPTrainer(HappyTrainer):
    """
    Trainer class for HappyWordPrediction
    """
    line_by_line = False

    def _tok_function(self, raw_dataset, dataclass_args: Union[WPTrainArgs, WPEvalArgs]):
        if not self.line_by_line:
            return preprocess_concatenate(tokenizer=self.tokenizer, dataset=raw_dataset,
                                      preprocessing_processes=dataclass_args.preprocessing_processes, mlm=True)
        else:
            def tokenize_function(example):
                return self.tokenizer(example["text"],
                                 add_special_tokens=True, truncation=True)

            tokenized_dataset = raw_dataset.map(tokenize_function, batched=True,
                                            num_proc=dataclass_args.preprocessing_processes,
                                            remove_columns=["text"])
            return tokenized_dataset

    def train(self, input_filepath, eval_filepath: str ="", dataclass_args: WPTrainArgs = WPTrainArgs()):
        """
        :param input_filepath: A file path to a text file that contains nothing but training data
        :param eval_filepath: todo
        :param dataclass_args: A WPTrainArgs() object
        :return: None
        """
        self.logger.info("Preprocessing training data...")
        train_data, eval_data = self._preprocess_data(input_filepath=input_filepath,
                                                      eval_filepath=eval_filepath,
                                                      dataclass_args=dataclass_args,
                                                      file_type="text")


        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        mlm_probability=dataclass_args.mlm_probability)
        self.logger.info("Training...")

        self._run_train(train_data, eval_data, dataclass_args, data_collator)


    def eval(self, input_filepath, dataclass_args: WPEvalArgs):
        """
        :param input_filepath: A file path to a text file that contains nothing but evaluating data
        :param dataclass_args: A WPEvalArgs() object
        :return: An EvalResult() object
        """
        dataset = load_dataset("text", data_files={"eval": input_filepath})

        self.line_by_line = dataclass_args.line_by_line

        tokenized_dataset = self._tok_function(dataset, dataclass_args)


        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        mlm_probability=dataclass_args.mlm_probability)
        result = self._run_eval(tokenized_dataset['eval'], data_collator, dataclass_args)

        return EvalResult(loss=result["eval_loss"])


    def test(self, input_filepath, solve, args):
        raise NotImplementedError()


