"""
This code is a modified version of the official documentation for the
transformer library by Hugging Face which can be found below.

We prioritized following the official documentation as close as possible to ensure we're using
robust methods. And also, to improve maintainability as they update the documentation.

https://huggingface.co/transformers/custom_datasets.html#sequence-classification-with-imdb-reviews"""

import csv
import torch
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from tqdm import tqdm


class TCTrainer(HappyTrainer):
    """
    A class for training text classification functionality
    """

    def train(self, input_filepath, args):
        contexts, labels = self._get_data(input_filepath)
        train_encodings = self.tokenizer(contexts, truncation=True, padding=True)
        train_dataset = TextClassificationDataset(train_encodings, labels)

        self._run_train(train_dataset, args)

    def eval(self, input_filepath):
        contexts, labels = self._get_data(input_filepath)
        eval_encodings = self.tokenizer(contexts, truncation=True, padding=True)
        eval_dataset = TextClassificationDataset(eval_encodings, labels)

        result = self._run_eval(eval_dataset)
        return EvalResult(loss=result["eval_loss"])

    def test(self, input_filepath, solve):
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
