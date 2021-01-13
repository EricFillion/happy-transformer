from .adaptor import Adaptor
from transformers import (
    BertForMaskedLM, BertTokenizerFast,
    BertForNextSentencePrediction, BertForQuestionAnswering,
    BertForSequenceClassification,

    DistilBertForMaskedLM, DistilBertTokenizerFast,
    DistilBertForSequenceClassification, DistilBertForQuestionAnswering,

    RobertaForMaskedLM, RobertaTokenizerFast,
    RobertaForQuestionAnswering, RobertaForSequenceClassification,
)

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

    def preprocess_mask_text(self, text:str)->str:
        print(text)
        return text.replace('[MASK]','<mask>')

    def postprocess_mask_prediction_token(self, text):
        return text[1:] if text[0] == "Ġ" else text