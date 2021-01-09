from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    DistilBertForMaskedLM,
    DistilBertTokenizerFast,
    FillMaskPipeline,

)
import torch
import collections
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.mwp.trainer import WPTrainer

WPOutput = collections.namedtuple("WPOutput", ["token_str", "score"])


class HappyWordPrediction(HappyTransformer):
    """
    A user facing class for text classification
    """
    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-uncased"):
        model = None
        tokenizer = None

        if model_type == "BERT":
            model = BertForMaskedLM.from_pretrained(model_name)
            tokenizer = BertTokenizerFast.from_pretrained(model_name)

        elif model_type == "DISTILBERT":
            model = DistilBertForMaskedLM.from_pretrained(model_name)
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

        super().__init__(model_type, model_name, model, tokenizer)
        device_number = 1 if torch.cuda.is_available() else -1
        self._pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer, device=device_number)
        self._trainer = WPTrainer(model, model_type, tokenizer, self._device, self.logger)

    def predict_mask(self, text, targets=None, top_k=1):
        """
        :param text: A string that contains the model's mask token
        :param targets: Optional. A list of strings of potential answers.
        All other answers will be ignored
        :param top_k: number of results. Default is 1
        :return: If top_k ==1: a dictionary with the keys "score" and "token_str"
                if  top_k >1: a list of dictionaries described above in order by score
        """
        if not isinstance(text, str):
            raise ValueError("the \"text\" argument must be a single string")

        result = self._pipeline(text, targets=targets, top_k=top_k)
        results = list()
        for answer in result:
            result = WPOutput(token_str=answer["token_str"], score=answer["score"])
            results.append(result)

        return results

    def train(self, input_filepath, args):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath):
        raise NotImplementedError("test() is currently not available")
