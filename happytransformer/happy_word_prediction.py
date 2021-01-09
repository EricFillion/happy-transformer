from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    DistilBertForMaskedLM,
    DistilBertTokenizerFast,
    FillMaskPipeline,

)
import torch

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.mwp.trainer import  MWPTrainer


class HappyWordPrediction(HappyTransformer):
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
        self._pipeline = FillMaskPipeline(model=model,
                                                    tokenizer=tokenizer, device=device_number)
        self._trainer = MWPTrainer(model,
                                  model_type, tokenizer, self._device, self.logger)
    def predict_masks(self, text, targets=None, top_k =1):
        """

        :param text: Either a single string, or strings that contains masks
        :param targets: Optional. A list of strings of potential answers. All other answers will be ignored
        :param top_k: number of results. Default is 1
        :return: If top_k ==1: a dictionary with the keys "score" and "token_str"
                if  top_k >1: a list of dictionaries described above in order by score
        """
        result = self._pipeline(text, targets=targets, top_k=top_k)

        if top_k ==1:
            result = result[0]
            del result['sequence']
            del result['token']
        else:
            for answer in result:
                del answer['sequence']
                del answer['token']
        return result



    def predict_mask(self):
        raise NotImplementedError()

    def train(self, input_filepath, args):
        self._trainer.train(input_filepath=input_filepath, args=args)

    def eval(self, input_filepath):
        return self._trainer.eval(input_filepath=input_filepath)

    def test(self, input_filepath):
        return self._trainer.test(input_filepath=input_filepath,pipeline=self._pipeline)
