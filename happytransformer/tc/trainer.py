"""
This code is a modified version of the official documentation for the
transformer library by Hugging Face which can be found below.

We prioritized following the official documentation as close as possible to ensure we're using
robust methods. And also, to improve maintainability as they update the documentation.

https://huggingface.co/transformers/custom_datasets.html#sequence-classification-with-imdb-reviews"""

from dataclasses import dataclass
import csv
import torch
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from transformers import DataCollatorWithPadding
from tqdm import tqdm
from happytransformer.tc.default_args import ARGS_TC_TRAIN, ARGS_TC_EVAL, ARGS_TC_TEST
import json

@dataclass
class TCTrainArgs:
    learning_rate: float = ARGS_TC_TRAIN["learning_rate"]
    num_train_epochs: int = ARGS_TC_TRAIN["num_train_epochs"]
    batch_size: int = ARGS_TC_TRAIN["batch_size"]
    weight_decay: float = ARGS_TC_TRAIN["weight_decay"]
    adam_beta1: float = ARGS_TC_TRAIN["adam_beta1"]
    adam_beta2: float = ARGS_TC_TRAIN["adam_beta2"]
    adam_epsilon: float = ARGS_TC_TRAIN["adam_epsilon"]
    max_grad_norm:  float = ARGS_TC_TRAIN["max_grad_norm"]
    save_preprocessed_data: bool = ARGS_TC_TRAIN["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_TC_TRAIN["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_TC_TRAIN["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_TC_TRAIN["load_preprocessed_data_path"]


@dataclass
class TCEvalArgs:
    save_preprocessed_data: bool = ARGS_TC_EVAL["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_TC_EVAL["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_TC_EVAL["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_TC_EVAL["load_preprocessed_data_path"]
    batch_size: int = ARGS_TC_EVAL["batch_size"]


@dataclass
class TCTestArgs:
    save_preprocessed_data: bool = ARGS_TC_TEST["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_TC_TEST["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_TC_TEST["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_TC_TEST["load_preprocessed_data_path"]

class TCTrainer(HappyTrainer):
    """
    A class for training text classification functionality
    """

    def train(self, input_filepath, dataclass_args: TCTrainArgs):

        if not dataclass_args.load_preprocessed_data:
            self.logger.info("Preprocessing dataset...")
            contexts, labels = self._get_data(input_filepath)
            train_encodings = self.tokenizer(contexts, truncation=True, padding=True)
        else:
            self.logger.info("Loading dataset from %s...", dataclass_args.load_preprocessed_data_path)
            train_encodings, labels = self._get_preprocessed_data(dataclass_args.load_preprocessed_data_path)

        if dataclass_args.save_preprocessed_data:
            self.logger.info("Saving training dataset to %s...", dataclass_args.save_preprocessed_data_path)
            input_ids = train_encodings["input_ids"]
            attention_mask = train_encodings["attention_mask"]
            self._generate_json(dataclass_args.save_preprocessed_data_path, input_ids, attention_mask, labels, "train")

        train_dataset = TextClassificationDataset(train_encodings, labels)
        data_collator = DataCollatorWithPadding(self.tokenizer)
        self._run_train(train_dataset, dataclass_args, data_collator)

    def eval(self, input_filepath, dataclass_args: TCEvalArgs):
        if not dataclass_args.load_preprocessed_data:
            self.logger.info("Preprocessing dataset...")
            contexts, labels = self._get_data(input_filepath)
            eval_encodings = self.tokenizer(contexts, truncation=True, padding=True)
        else:
            self.logger.info("Loading dataset from %s...", dataclass_args.load_preprocessed_data_path)
            eval_encodings, labels = self._get_preprocessed_data(dataclass_args.load_preprocessed_data_path)

        if dataclass_args.save_preprocessed_data:
            self.logger.info("Saving training dataset to %s...", dataclass_args.save_preprocessed_data_path)
            input_ids = eval_encodings["input_ids"]
            attention_mask = eval_encodings["attention_mask"]
            self._generate_json(dataclass_args.save_preprocessed_data_path, input_ids, attention_mask, labels, "train")


        eval_dataset = TextClassificationDataset(eval_encodings, labels)
        data_collator = DataCollatorWithPadding(self.tokenizer)

        result = self._run_eval(eval_dataset, data_collator, dataclass_args)
        return EvalResult(loss=result["eval_loss"])

    def test(self, input_filepath, solve, dataclass_args: TCTestArgs):
        """
        See docstring in HappyQuestionAnswering.test()
        solve: HappyQuestionAnswering.answers_to_question()
        """
        contexts = self._get_data(input_filepath, test_data=True)

        return [
            solve(context)
            for context in tqdm(contexts)
        ]

    @staticmethod
    def _get_data(filepath, test_data=False):
        """
        Used for parsing data for training and evaluating (both contain labels)
        :param filepath: a string that contains the location of the data
        :return:
        """
        contexts = []
        labels = []
        with open(filepath, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                contexts.append(row['text'])
                if not test_data:
                    labels.append(int(row['label']))
        csv_file.close()

        if not test_data:
            return contexts, labels
        return contexts

    @staticmethod
    def _generate_json(json_path, input_ids, attention_mask, labels, name):
        data = {}
        data[name] = []
        data = {
            name: [
                {
                    'attention_mask': input_id,
                    'input_ids': attention_mask,
                    'labels': label
                }
                for input_id, attention_mask, label in zip(input_ids, attention_mask, labels)
            ]
        }

        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)

    @staticmethod
    def _get_preprocessed_data(filepath):
        """
        Used for Fetching preprocessed data)
        :param filepath: a string that contains the location of the data
        :return:
        """

        # dataset = load_dataset("csv", data_files={"train": filepath})
        input_ids = []
        attention_mask = []
        labels = []
        with open(filepath) as json_file:
            data = json.load(json_file)
        json_file.close()

        for case in data["train"]:
            input_ids.append(case['input_ids'])
            attention_mask.append(case['attention_mask'])
            labels.append(case['labels'])


        train_encodings = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        return train_encodings, labels


class TextClassificationDataset(torch.utils.data.Dataset):
    """
    A class to allow the training and testing data to be used by
    a transformers.Trainer object
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class TextClassificationDatasetTest(torch.utils.data.Dataset):
    """
    A class to allow the testing data to be used by
    a transformers.Trainer object
    """
    def __init__(self, encodings, length):
        self.encodings = encodings
        self.length = length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.length
