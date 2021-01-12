from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    AlbertForMaskedLM,
    AlbertTokenizerFast,
    DistilBertForMaskedLM,
    DistilBertTokenizerFast,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    FillMaskPipeline,
)
import torch
from collections import namedtuple
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.mwp.trainer import WPTrainer
from happytransformer.cuda_detect import detect_cuda_device_number

WordPredictionResult = namedtuple("WordPredictionResult", ["token_str", "score"])


class HappyWordPrediction(HappyTransformer):
    """
    A user facing class for text classification
    """
    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-uncased"):
        model = None
        tokenizer = None

        if model_type == "ALBERT":
            model = AlbertForMaskedLM.from_pretrained(model_name)
            tokenizer = AlbertTokenizerFast.from_pretrained(model_name)
        elif model_type == "BERT":
            model = BertForMaskedLM.from_pretrained(model_name)
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
        elif model_type == "DISTILBERT":
            model = DistilBertForMaskedLM.from_pretrained(model_name)
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        elif model_type == "ROBERTA":
            model = RobertaForMaskedLM.from_pretrained(model_name)
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        else:
            raise ValueError(self.model_type_error)
        super().__init__(model_type, model_name, model, tokenizer)

        device_number = detect_cuda_device_number()

        self._pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer, device=device_number)

        self._trainer = WPTrainer(model, model_type, tokenizer, self._device, self.logger)

    def predict_mask(self, text, targets=None, top_k=1):
        """
        :param text: A string that contains the model's mask token
        :param targets: Optional. A list of strings of potential answers.
        All other answers will be ignored
        :param top_k: number of results. Default is 1
        :return: A named WordPredictionResult Named Tuple with the following keys: token_str and score
        """
        if not isinstance(text, str):
            raise ValueError("the \"text\" argument must be a single string")

        if self.model_type == "ROBERTA":
            text = text.replace("[MASK]", "<mask>")

        result = self._pipeline(text, targets=targets, top_k=top_k)

        if self.model_type == "ALBERT":
            for answer in result:
                if answer["token_str"][0] == "▁":
                    answer["token_str"] = answer["token_str"][1:]
        elif self.model_type == "ROBERTA":
            for answer in result:
                if answer["token_str"][0] == "Ġ":
                    answer["token_str"] = answer["token_str"][1:]
        results = [
            WordPredictionResult(
                token_str=answer["token_str"], 
                score=answer["score"]
            )
            for answer in result
        ]
        
        return results

    def train(self, input_filepath, args):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath):
        raise NotImplementedError("test() is currently not available")
