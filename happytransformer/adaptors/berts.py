from .adaptor import Adaptor
from transformers import (
    BertForMaskedLM, BertTokenizerFast,
    BertForNextSentencePrediction, BertForQuestionAnswering,
    BertForSequenceClassification, BertForTokenClassification,

    DistilBertForMaskedLM, DistilBertTokenizerFast,
    DistilBertForSequenceClassification, DistilBertForQuestionAnswering,
    DistilBertForTokenClassification,

    RobertaForMaskedLM, RobertaTokenizerFast,
    RobertaForQuestionAnswering, RobertaForSequenceClassification,
    RobertaForTokenClassification,

    AlbertForMaskedLM, AlbertTokenizerFast, AlbertForQuestionAnswering,
    AlbertForSequenceClassification,     AlbertForTokenClassification,
)

class BertAdaptor(Adaptor):
    Tokenizer = BertTokenizerFast
    MaskedLM = BertForMaskedLM
    NextSentencePrediction = BertForNextSentencePrediction
    QuestionAnswering = BertForQuestionAnswering
    SequenceClassification = BertForSequenceClassification
    TokenClassification = BertForTokenClassification

class DistilBertAdaptor(Adaptor):
    Tokenizer = DistilBertTokenizerFast
    MaskedLM = DistilBertForMaskedLM
    QuestionAnswering = DistilBertForQuestionAnswering
    SequenceClassification = DistilBertForSequenceClassification
    TokenClassification = DistilBertForTokenClassification


class RobertaAdaptor(Adaptor):
    Tokenizer = RobertaTokenizerFast
    MaskedLM = RobertaForMaskedLM
    QuestionAnswering = RobertaForQuestionAnswering
    SequenceClassification = RobertaForSequenceClassification
    TokenClassification = RobertaForTokenClassification

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
    TokenClassification = AlbertForTokenClassification

    @staticmethod
    def postprocess_mask_prediction_token(text):
        return text[1:] if text[0] == "▁" else text