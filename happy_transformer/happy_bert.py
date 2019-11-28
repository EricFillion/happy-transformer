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
        Gets the softmaxes of the predictions for each index in the the given
        input string.
        Returned tensor will be in shape:
            [1, <tokens in string>, <possible options for token>]

        :param text: a tokenized string to be used by the transformer.
        :return: a tensor of the softmaxes of the predictions of the
                 transformer
        """
        segments_ids = self._get_segment_ids(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(text)


        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])


        with torch.no_grad():
            outputs = self.transformer(tokens_tensor,
                                       token_type_ids=segments_tensors)
            predictions = outputs[0]
            softmax = self._softmax(predictions)

            return softmax
