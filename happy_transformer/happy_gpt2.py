# pylint: disable=W0511
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from happy_transformer.happy_transformer import HappyTransformer


class HappyGPT2(HappyTransformer):

    def __init__(self, model='gpt2'):
        super().__init__()

        self.transformer = GPT2LMHeadModel.from_pretrained(model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.mask_token = '<mask>'
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = ''
        self.cls_token = ''

        self.model = 'GPT2'

        self.transformer.eval()

    def _get_segment_ids(self, tokenized_text):
        return [0] * len(tokenized_text)
