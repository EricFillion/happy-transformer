import torch

from collections import namedtuple
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.mwp.trainer import WPTrainer

from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    AlbertForMaskedLM,
    AlbertTokenizerFast,
    DistilBertForMaskedLM,
    DistilBertTokenizerFast,
    FillMaskPipeline,
)

from happytransformer.happy_transformer import HappyTransformer

NextSentenceResult = namedtuple("NextSentenceResult", ("next_sentence", "score"))


class HappyNextSentence(HappyTransformer):
    """
    A user facing class for next sentence prediction
    """
    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-uncased"):

        if model_type == "ALBERT":
            model = AlbertForMaskedLM.from_pretrained(model_name)
            tokenizer = AlbertTokenizerFast.from_pretrained(model_name)

        elif model_type == "BERT":
            model = BertForMaskedLM.from_pretrained(model_name)
            tokenizer = BertTokenizerFast.from_pretrained(model_name)

        elif model_type == "DISTILBERT":
            model = DistilBertForMaskedLM.from_pretrained(model_name)
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        else:
            raise ValueError(self.model_type_error)
        super().__init__(model_type, model_name, model, tokenizer)
        device_number = 1 if torch.cuda.is_available() else -1
        self._pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer, device=device_number)
        self._trainer = WPTrainer(model, model_type, tokenizer, self._device, self.logger)

    def predict_next_sentence(self):
        """
        TODO: Create DocString
        """
        raise NotImplementedError()

    def train(self, input_filepath, args):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath):
        raise NotImplementedError("test() is currently not available")
