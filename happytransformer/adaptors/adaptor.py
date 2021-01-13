from typing import Type
from transformers import (
    PreTrainedModel,
    BertForMaskedLM, BertTokenizerFast,
    RobertaForMaskedLM, RobertaTokenizerFast
)
from transformers.tokenization_utils import PreTrainedTokenizerBase

class Adaptor:
    @property
    def tokenizer(self)->Type[PreTrainedTokenizerBase]:
        raise NotImplementedError()

    @property
    def masked_language_model(self)->Type[PreTrainedModel]:
        raise NotImplementedError()

    def preprocess_text(self, text:str)->str:
        return text

    def postprocess_token(self, text:str)->str:
        return text

class BertAdaptor(Adaptor):
    @property
    def tokenizer(self):
        return BertTokenizerFast
    @property
    def get_masked_language_model(self):
        return BertForMaskedLM

class RobertaAdaptor(Adaptor):
    @property
    def tokenizer(self):
        return RobertaTokenizerFast
    @property
    def masked_language_model(self):
        return RobertaForMaskedLM

    def preprocess_text(self, text:str)->str:
        print(text)
        return text.replace('[MASK]','<mask>')

    def postprocess_token(self, text):
        return (
            text[1:] 
            if text[0] == "Ġ" 
            else text
        )

ADAPTORS = {
    'BERT':BertAdaptor(),
    'ROBERTA':RobertaAdaptor()
}

def get_adaptor(model_type:str)->Adaptor:
    if model_type in ADAPTORS:
        return ADAPTORS[model_type]
    else:
        raise ValueError(f'Model type <{model_type}> not currently supported')