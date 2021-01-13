from typing import Type
from transformers import (
    PreTrainedModel,

    BertForMaskedLM, BertTokenizerFast,
    BertForNextSentencePrediction, BertForQuestionAnswering,
    BertForSequenceClassification,

    RobertaForMaskedLM, RobertaTokenizerFast,
    RobertaForQuestionAnswering, RobertaForSequenceClassification,

    AlbertForMaskedLM, AlbertTokenizerFast, AlbertForQuestionAnswering,
    AlbertForSequenceClassification,

    DistilBertForMaskedLM, DistilBertTokenizerFast,
    DistilBertForQuestionAnswering, DistilBertForSequenceClassification
)
from transformers.tokenization_utils import PreTrainedTokenizerBase

class Adaptor:
    '''
    Holds a few functions for implementation details.
    Does NOT store any state.
    '''
    @property
    def Tokenizer(self)->Type[PreTrainedTokenizerBase]:
        raise NotImplementedError()

    @property
    def MaskedLM(self)->Type[PreTrainedModel]:
        raise NotImplementedError()

    @property
    def NextSentencePrediction(self)->Type[PreTrainedModel]:
        raise NotImplementedError()

    @property
    def QuestionAnswering(self)->Type[PreTrainedModel]:
        raise NotImplementedError()

    @property
    def SequenceClassification(self)->Type[PreTrainedModel]:
        raise NotImplementedError()

    def preprocess_text(self, text:str)->str:
        return text

    def postprocess_token(self, text:str)->str:
        return text

class BertAdaptor(Adaptor):
    @property
    def Tokenizer(self):
        return BertTokenizerFast
    @property
    def MaskedLM(self):
        return BertForMaskedLM
    @property
    def NextSentencePrediction(self):
        return BertForNextSentencePrediction
    @property
    def QuestionAnswering(self):
        return BertForQuestionAnswering
    @property
    def SequenceClassification(self):
        return BertForSequenceClassification

class DistilBertAdaptor(Adaptor):
    @property
    def Tokenizer(self):
        return DistilBertTokenizerFast
    @property
    def MaskedLM(self):
        return DistilBertForMaskedLM
    @property
    def QuestionAnswering(self):
        return DistilBertForQuestionAnswering
    @property
    def SequenceClassification(self):
        return DistilBertForSequenceClassification

class RobertaAdaptor(Adaptor):
    @property
    def Tokenizer(self):
        return RobertaTokenizerFast
    @property
    def MaskedLM(self):
        return RobertaForMaskedLM
    @property
    def QuestionAnswering(self):
        return RobertaForQuestionAnswering
    @property
    def SequenceClassification(self):
        return RobertaForSequenceClassification

    def preprocess_text(self, text:str)->str:
        print(text)
        return text.replace('[MASK]','<mask>')

    def postprocess_token(self, text):
        return text[1:] if text[0] == "Ġ" else text

class AlbertAdaptor(Adaptor):
    @property
    def Tokenizer(self):
        return AlbertTokenizerFast
    @property
    def MaskedLM(self):
        return AlbertForMaskedLM
    @property
    def QuestionAnswering(self):
        return AlbertForQuestionAnswering
    @property
    def SequenceClassification(self):
        return AlbertForSequenceClassification
    
    def postprocess_token(self, text):
        return text[1:] if text[0] == "▁" else text

ADAPTORS = {
    'BERT':BertAdaptor(),
    'DISTILBERT':DistilBertAdaptor(),
    'ROBERTA':RobertaAdaptor(),
    'ALBERT':AlbertAdaptor()
}

def get_adaptor(model_type:str)->Adaptor:
    if model_type in ADAPTORS:
        return ADAPTORS[model_type]
    else:
        raise ValueError(f'Model type <{model_type}> not currently supported')