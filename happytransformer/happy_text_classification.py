"""
Contains a class called HappyTextClassification that performs text classification
"""
from dataclasses import dataclass

import torch
from transformers import TextClassificationPipeline, AutoConfig

from happytransformer.tc.trainer import TCTrainer
from happytransformer.cuda_detect import detect_cuda_device_number
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.adaptors import get_adaptor
from happytransformer.tc.default_args import ARGS_TC_TRAIN

@dataclass
class TextClassificationResult:
    label: str
    score: float

class HappyTextClassification(HappyTransformer):
    """
    A user facing class for Text Classification
    """

    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-uncased", num_labels=2):
        self.adaptor = get_adaptor(model_type)
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)

        model = self.adaptor.SequenceClassification.from_pretrained(model_name, config=config)
        tokenizer = self.adaptor.Tokenizer.from_pretrained(model_name)

        super().__init__(model_type, model_name, model, tokenizer)

        device_number = detect_cuda_device_number()
        self._pipeline = TextClassificationPipeline(
            model=model, tokenizer=tokenizer, 
            device=device_number
        )

        self._trainer = TCTrainer(
            self._model, self.model_type, 
            self._tokenizer, self._device, self.logger
        )

    def classify_text(self, text: str) -> TextClassificationResult:
        """
        Classify text to a label based on model's training
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
