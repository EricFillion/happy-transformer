"""
HappyBERT
"""

# disable pylint TODO warning
# pylint: disable=W0511
import torch
from transformers import BertForMaskedLM, BertTokenizer

from happy_transformer.happy_transformer import HappyTransformer


class HappyBERT(HappyTransformer):
    """
    A wrapper over PyTorch's BERT transformer implementation
    """

    def __init__(self, model='bert-large-uncased'):
        super().__init__()
        self.transformer = BertForMaskedLM.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.model = 'BERT'
        self.transformer.eval()

    def _get_prediction_softmax(self, text: str):
        """
        BERT's "_get_prediction_softmax" is the default in HappyTransformer
        """
        return super()._get_prediction_softmax(text)