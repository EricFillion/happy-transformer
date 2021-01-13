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
from dataclasses import dataclass
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.mwp.trainer import WPTrainer
from happytransformer.cuda_detect import detect_cuda_device_number
from happytransformer.adaptors.adaptor import get_adaptor
from typing import List

@dataclass
class WordPredictionResult:
    token:str
    score:float

class HappyWordPrediction(HappyTransformer):
    """
    A user facing class for text classification
    """
    def __init__(self, model_type:str="DISTILBERT",
                 model_name:str="distilbert-base-uncased"):
        adaptor = get_adaptor(model_type)
        model = adaptor.get_masked_language_model(model_name)
        tokenizer = adaptor.get_tokenizer(model_name)

        super().__init__(model_type, model_name, model, tokenizer)

        device_number = detect_cuda_device_number()

        self._pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer, device=device_number)

        self._trainer = WPTrainer(model, model_type, tokenizer, self._device, self.logger)

    def predict_mask(self, 
        text:str, targets:List[str]=None, top_k:int=1
    ) -> List[WordPredictionResult]:
        """
        Predict [MASK] tokens in a string.
        targets limit possible guesses if supplied.
        top_k describes number of targets to return*
        *top_k does not apply if targets is supplied
        """
        if not isinstance(text, str):
            raise ValueError("the \"text\" argument must be a single string")

        if self.model_type == "ROBERTA":
            text = text.replace("[MASK]", "<mask>")

        answers = self._pipeline(text, targets=targets, top_k=top_k)

        if self.model_type == "ALBERT":
            for answer in answers:
                if answer["token_str"][0] == "▁":
                    answer["token_str"] = answer["token_str"][1:]
        elif self.model_type == "ROBERTA":
            for answer in answers:
                if answer["token_str"][0] == "Ġ":
                    answer["token_str"] = answer["token_str"][1:]
        return [
            WordPredictionResult(
                token=answer["token_str"], 
                score=answer["score"]
            )
            for answer in answers
        ]

    def train(self, input_filepath, args):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath):
        raise NotImplementedError("test() is currently not available")
