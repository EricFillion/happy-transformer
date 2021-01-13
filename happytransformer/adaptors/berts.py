from .adaptor import Adaptor
from transformers import (
    BertForMaskedLM, BertTokenizerFast,
    BertForNextSentencePrediction, BertForQuestionAnswering,
    BertForSequenceClassification,

    DistilBertForMaskedLM, DistilBertTokenizerFast,
    DistilBertForSequenceClassification, DistilBertForQuestionAnswering,

    RobertaForMaskedLM, RobertaTokenizerFast,
    RobertaForQuestionAnswering, RobertaForSequenceClassification,

    AlbertForMaskedLM, AlbertTokenizerFast, AlbertForQuestionAnswering,
    AlbertForSequenceClassification
)

class BertAdaptor(Adaptor):
    Tokenizer = BertTokenizerFast
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

    @staticmethod
    def preprocess_mask_text(text):
        return text.replace('[MASK]', '<mask>')

    @staticmethod
    def postprocess_mask_prediction_token(text):
        return text[1:] if text[0] == "Ġ" else text

class AlbertAdaptor(Adaptor):
    Tokenizer = AlbertTokenizerFast
    MaskedLM = AlbertForMaskedLM
    QuestionAnswering = AlbertForQuestionAnswering
    SequenceClassification = AlbertForSequenceClassification
    
    @staticmethod
    def postprocess_mask_prediction_token(text):
        return text[1:] if text[0] == "▁" else text