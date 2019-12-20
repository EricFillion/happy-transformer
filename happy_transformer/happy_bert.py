"""
HappyBERT
"""

# disable pylint TODO warning
# pylint: disable=W0511
import torch
from transformers import BertForMaskedLM, BertForNextSentencePrediction, BertTokenizer

from happy_transformer.happy_transformer import HappyTransformer


class HappyBERT(HappyTransformer):
    """
    A wrapper over PyTorch's BERT transformer implementation
    """

    def __init__(self, model='bert-large-uncased', initial_transformers=[]):
        super().__init__(model, initial_transformers)
        self.mlm = None  # Masked Language Model
        self.nsp = None  # Next Sentence Prediction
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.model = 'BERT'

    def _get_masked_language_model(self):
        """
        Initializes the BertForMaskedLM transformer
        """
        self.mlm = BertForMaskedLM.from_pretrained(self.model_to_use)
        self.mlm.eval()

    def _get_next_sentence_prediction(self):
        """
        Initializes the BertForNextSentencePrediction transformer
        """
        self.nsp = BertForNextSentencePrediction.from_pretrained(self.model_to_use)
        self.nsp.eval()

    def _get_prediction_softmax(self, text: str):
        """
        BERT's "_get_prediction_softmax" is the default in HappyTransformer
        """
        return super()._get_prediction_softmax(text)