"""
Contains a class called HappyTextClassification that performs text classification
"""
from collections import namedtuple
import torch

from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    AlbertForSequenceClassification,
    AlbertTokenizerFast,
AutoConfig,
TextClassificationPipeline
)
from happytransformer.tc.trainer import TCTrainer

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.tc.default_args import ARGS_TC_TRAIN


TextClassificationResult = namedtuple("TextClassificationResult", ["label", "score"])

class HappyTextClassification(HappyTransformer):
    """
    A user facing class for Text Classification
    """

    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-uncased", num_labels=2):
        model = None
        tokenizer = None
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)

        if model_type == "ALBERT":
            model = AlbertForSequenceClassification.from_pretrained(model_name, config=config)
            tokenizer = AlbertTokenizerFast.from_pretrained(model_name)
        elif model_type == "BERT":
            model = BertForSequenceClassification.from_pretrained(model_name, config=config)
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
        elif model_type == "DISTILBERT":
            model = DistilBertForSequenceClassification.from_pretrained(model_name, config=config)
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

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
        :param text: A text string to be classified
        :return: A dictionary with keys: label and score,
        """
        # Blocking allowing a for a list of strings
        if not isinstance(text, str):
            raise ValueError("the \"text\" argument must be a single string")
        results = self._pipeline(text)
        # we do not support predicting a list of  texts, so only first prediction is relevant
        first_result = results[0]

        return TextClassificationResult(label=first_result["label"], score=first_result["score"])


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
        return self._trainer.test(input_filepath=input_filepath, solve=self.classify_text)
