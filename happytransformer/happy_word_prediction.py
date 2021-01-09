from happytransformer.qa.trainer import QATrainer

from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    AlbertForMaskedLM,
    AlbertTokenizerFast,
    DistilBertForMaskedLM,
    DistilBertTokenizerFast,
    FillMaskPipeline,

)
import torch

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.mwp.trainer import  MWPTrainer


class HappyWordPrediction(HappyTransformer):
    def __init__(self, model_type="BERT",
                 model_name="bert-large-uncased-whole-word-masking-finetuned-squad", device=None):
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

        else:
            raise ValueError("model_type must be BERT, DISTILBERT or ALBERT")


        super().__init__(model_type, model_name, model, tokenizer)
        device_number = 1 if torch.cuda.is_available() else -1
        self._pipeline = FillMaskPipeline(model=model,
                                                    tokenizer=tokenizer, device=device_number)
        self._trainer = MWPTrainer(model,
                                  model_type, tokenizer, self._device, self.logger)
    def predict_masks(self):
        raise NotImplementedError()

    def predict_mask(self):
        raise NotImplementedError()

    def train(self, input_filepath, args):
        self._trainer.train(input_filepath=input_filepath, args=args)

    def eval(self, input_filepath):
        return self._trainer.eval(input_filepath=input_filepath)

    def test(self, input_filepath):
        return self._trainer.test(input_filepath=input_filepath,)
