from transformers import (
    AlbertForMaskedLM, AlbertTokenizerFast, AlbertForQuestionAnswering,
    AlbertForSequenceClassification
)
from .adaptor import Adaptor

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
    
    def postprocess_mask_prediction_token(self, text):
        return text[1:] if text[0] == "▁" else text