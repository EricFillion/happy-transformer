from happytransformer.tc.trainer import TCTrainer

from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast
)

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.tc.default_args import ARGS_TC_TRAIN
import numpy as np

from happytransformer.util import softmax_of_matrix


class HappyTextClassification(HappyTransformer):

    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-uncased-finetuned-sst-2-english", device=None):
        model = None
        tokenizer = None

        if model_type == "BERT":
            model = BertForSequenceClassification.from_pretrained(model_name)
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
        elif model_type == "DISTILBERT":
            model = DistilBertForSequenceClassification.from_pretrained(model_name)
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

        super().__init__(model_type, model_name, model, tokenizer, device)

        self._trainer = TCTrainer(self._model,
                                  self.model_type, self._tokenizer, self._device, self.logger)

    def classify_text(self, text):
        """
        :param text: A text string to be classified
        :return:
        """

        inputs = self._tokenizer(text, return_tensors="pt")
        output = self._model(**inputs)
        logits = output.logits
        scores = logits.detach().cpu()
        softmax = softmax_of_matrix(scores)[0]
        preds = np.argmax(scores.numpy(), axis=1)
        return {
            "answer": preds[0],
            'softmax': softmax
        }

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
        return self._trainer.test(input_filepath=input_filepath)
