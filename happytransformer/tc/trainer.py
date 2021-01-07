"""
This code is a modified version of the official documentation for the
transformer library by Hugging Face which can be found below.

We prioritized following the official documentation as close as possible to ensure we're using
robust methods. And also, to improve maintainability as they update the documentation.

https://huggingface.co/transformers/custom_datasets.html#sequence-classification-with-imdb-reviews"""

import csv
import torch
from torch.utils.data import DataLoader

from transformers import Trainer

from happytransformer.happy_trainer import HappyTrainer

class TCTrainer(HappyTrainer):

    def __init__(self, model, model_type, tokenizer, device, logger):
        super().__init__(model, model_type, tokenizer, device, logger)

    def train(self, input_filepath,  output_path, args):
        contexts, labels = self.__get_data(input_filepath)
        train_encodings = self.tokenizer(contexts, truncation=True, padding=True)
        train_dataset = TextClassificationDataset(train_encodings, labels)

        training_args = self._get_training_args(args, output_path)

        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
        )
        trainer.train()

    def eval(self, input_filepath, output_path):
        contexts, labels = self.__get_data(input_filepath)
        eval_encodings = self.tokenizer(contexts, truncation=True, padding=True)

        eval_dataset = TextClassificationDataset(eval_encodings, labels)
        eval_args = self._get_test_eval_args(output_path)

        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=eval_args,
            eval_dataset=eval_dataset,  # training dataset

        )

        return trainer.evaluate()


    def test(self, input_filepath, output_path):
        contexts = self.__get_data(input_filepath, True)
        test_encodings = self.tokenizer(contexts, truncation=True, padding=True)

        test_dataset = TextClassificationDatasetTest(test_encodings, len(contexts))
        test_args = self._get_test_eval_args(output_path)

        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=test_args
        )
        print(trainer.predict(test_dataset))


    @staticmethod
    def __get_data(filepath, test_data=False):
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
    def __init__(self, encodings, length):
        self.encodings = encodings
        self.length = length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.length