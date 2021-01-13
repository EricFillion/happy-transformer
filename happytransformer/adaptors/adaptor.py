from transformers import (
    PreTrainedModel,
    BertForMaskedLM, BertTokenizerFast
)
from transformers.tokenization_utils import PreTrainedTokenizerBase

class Adaptor:
    def get_tokenizer(self, model_name:str)->PreTrainedTokenizerBase:
        raise NotImplementedError()

    def get_masked_language_model(self, model_name:str)->PreTrainedModel:
        raise NotImplementedError()

    def preprocess_text(self, text:str)->str:
        return text

class BERTAdaptor(Adaptor):
    def get_tokenizer(self,  model_name:str):
        return BertTokenizerFast.from_pretrained(model_name)

    def get_masked_language_model(self, model_name:str):
        return BertForMaskedLM.from_pretrained(model_name)

ADAPTORS = {
    'BERT':BERTAdaptor()
}

def get_adaptor(model_type:str)->Adaptor:
    if model_type in ADAPTORS:
        return ADAPTORS[model_type]
    else:
        raise ValueError(f'Model type <{model_type}> not currently supported')