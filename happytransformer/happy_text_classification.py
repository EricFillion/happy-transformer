"""
Contains a class called HappyTextClassification that performs text classification
"""

import torch

from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    AlbertForSequenceClassification,
    AlbertTokenizerFast,


    TextClassificationPipeline
)
from happytransformer.tc.trainer import TCTrainer

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.tc.default_args import ARGS_TC_TRAIN


class HappyTextClassification(HappyTransformer):
    """
    A user facing class for Text Classification
    """

    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        model = None
        tokenizer = None

        if model_type == "BERT":
            model = BertForSequenceClassification.from_pretrained(model_name)
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
        elif model_type == "DISTILBERT":
            model = DistilBertForSequenceClassification.from_pretrained(model_name)
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        elif model_type == "ALBERT":
            model = AlbertForSequenceClassification.from_pretrained(model_name)
            tokenizer = AlbertTokenizerFast.from_pretrained(model_name)
        else:
            raise ValueError(self.model_type_error)

        super().__init__(model_type, model_name, model, tokenizer)

        device_number = 1 if torch.cuda.is_available() else -1
        # from documentation " a positive will run the model on the associated CUDA device id."
        # todo: get device ID if torch.cuda.is_available()

        self._pipeline = TextClassificationPipeline(model=model,
                                                    tokenizer=tokenizer, device=device_number)


        self._trainer = TCTrainer(self._model,
                                  self.model_type, self._tokenizer, self._device, self.logger)

    def classify_text(self, text):
        """
        :param text: A text string to be classified, or a list of strings
        :return: either a single dictionary with keys: label and score,
        or a list of these dictionaries with the same keys
        """
        return self._pipeline(text)


    def train(self, input_filepath, args=ARGS_TC_TRAIN):
        """
        Trains the question answering model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values: text,
         label

        args: a dictionary that contains settings found under

        return: None

        """
        self._trainer.train(input_filepath=input_filepath, args=args)

    def eval(self, input_filepath):
        """
        Evaluated the text classification answering model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
         text, label

        return: #todo
        """
        return self._trainer.eval(input_filepath=input_filepath)

    def test(self, input_filepath):
        """
        Tests the text classification  model. Used to obtain results

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header value:
         text
        return: #todo
        """
        return self._trainer.test(input_filepath=input_filepath, pipeline=self._pipeline)
