from transformers import (
    AlbertForMaskedLM, AlbertTokenizerFast, AlbertForQuestionAnswering,
    AlbertForSequenceClassification
)
from .adaptor import Adaptor

class AlbertAdaptor(Adaptor):
    Tokenizer = AlbertTokenizerFast
    MaskedLM = AlbertForMaskedLM
    QuestionAnswering = AlbertForQuestionAnswering
    SequenceClassification = AlbertForSequenceClassification
    
    def postprocess_mask_prediction_token(self, text):
        return text[1:] if text[0] == "▁" else text