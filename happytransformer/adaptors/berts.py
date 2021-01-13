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
    Tokenizer =  BertTokenizerFast
    MaskedLM = BertForMaskedLM
    NextSentencePrediction = BertForNextSentencePrediction
    QuestionAnswering = BertForQuestionAnswering
    SequenceClassification = BertForSequenceClassification

class DistilBertAdaptor(Adaptor):
    Tokenizer = DistilBertTokenizerFast
    MaskedLM = DistilBertForMaskedLM
    QuestionAnswering = DistilBertForQuestionAnswering
    SequenceClassification = DistilBertForSequenceClassification

class RobertaAdaptor(Adaptor):
    Tokenizer = RobertaTokenizerFast
    MaskedLM = RobertaForMaskedLM
    QuestionAnswering = RobertaForQuestionAnswering
    SequenceClassification = RobertaForSequenceClassification

    def preprocess_mask_text(self, text:str)->str:
        print(text)
        return text.replace('[MASK]','<mask>')

    def postprocess_mask_prediction_token(self, text):
        return text[1:] if text[0] == "Ġ" else text