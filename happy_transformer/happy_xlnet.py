from transformers import XLNetLMHeadModel, XLNetTokenizer

from happy_transformer.happy_transformer import HappyTransformer
from happy_transformer.sequence_classification import SequenceClassifier
from happy_transformer.classifier_utils import classifier_args
import logging


import os

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)




from happy_transformer.classifier_utils import convert_examples_to_features, processors


class HappyXLNET(HappyTransformer):
    """
    Implementation of XLNET for masked word prediction
    """

    def __init__(self, model='xlnet-base-cased', initial_transformers=[]):
        super().__init__(model, initial_transformers)
        self.mlm = None
        self.tokenizer = XLNetTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.model = 'XLNET'
        self.classifier_name = ""

    def _get_masked_language_model(self):
        """
        Initializes the XLNetLMHeadModel transformer
        """
        self.mlm = XLNetLMHeadModel.from_pretrained(self.model_to_use)
        self.mlm.eval()


