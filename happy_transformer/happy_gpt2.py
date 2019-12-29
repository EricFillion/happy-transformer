"""
We will finish HappyGPT2 in January 2020
# pylint: disable=W0511
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from happy_transformer.happy_transformer import HappyTransformer


class HappyGPT2(HappyTransformer):

    def __init__(self, model='gpt2', initial_transformers=[]):
        super().__init__(model)
        self.transformer = GPT2LMHeadModel.from_pretrained(model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.mask_token = '<mask>'
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = ''
        self.cls_token = ''
        self.model = 'GPT2'
        self._get_initial_transformers(initial_transformers)

    def _get_masked_language_model(self):
        \"""
        Initializes the GPT2LMHeadModel transformer
        \"""
        self.mlm = GPT2LMHeadModel.from_pretrained(self.model_to_use)
        self.mlm.eval()

    def _get_segment_ids(self, tokenized_text):
        return [0] * len(tokenized_text)
"""