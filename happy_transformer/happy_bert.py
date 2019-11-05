# disable pylint TODO warning
# pylint: disable=W0511

import torch
from transformers import BertForMaskedLM, BertTokenizer
import numpy as np

from happy_transformer.happy_transformer import HappyTransformer


class HappyBERT(HappyTransformer):

    def __init__(self, model='bert-large-uncased'):
        super().__init__()
        self.transformer = BertForMaskedLM.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer._mask_token
        self.sep_token = self.tokenizer._sep_token
        self.cls_token = self.tokenizer._cls_token

        self.model = 'BERT'

        self.transformer.eval()

    def predict_mask(self, text: str):
        """
        :param text: a string with a masked token within it
        :return: predicts the most likely word to fill the mask and its score
        """

        # TODO: put in HappyBERT. Overwrite HappyTransformer.
        # TODO: easy: create a method to check if the sentence is valid
        # TODO: easy: if the sentence is not valid, provide the user with input requirements
        # TODO: easy: if sentence is not valid, indicate where the user messed up

        # TODO: medium: make it more modular

        tokens_tensor, segments_tensors, masked_index =\
            self._HappyTransformer__get_tensors_and_mask_idx(text)

        with torch.no_grad():
            outputs = self.transformer(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

            softmax = self._HappyTransformer__softmax(predictions)

            top_prediction = torch.topk(softmax[0, masked_index], 1)
            prediction_softmax = top_prediction[0].tolist()
            prediction_index = top_prediction[1].tolist()

            prediction_token = self.tokenizer.convert_ids_to_tokens(prediction_index)

           # TODO: easy: del various variables

            if self.gpu_support == "cuda":
                torch.cuda.empty_cache()

            return prediction_token[0], prediction_softmax

    def predict_mask_with_options(self, text: str, options: list):

        tokens_tensor, segments_tensors, masked_index =\
            self._HappyTransformer__get_tensors_and_mask_idx(text)

        with torch.no_grad():
            outputs = self.transformer(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

            softmax = self._HappyTransformer__softmax(predictions)[0]

            option_ids = [self.tokenizer.encode(option) for option in options]

            option_probs = list(map(lambda x: self.soft_sum(x, softmax, masked_index), option_ids))
            tupled_option = tuple(zip(options, option_probs))
            ranked_scores = sorted(tupled_option, key=lambda x: x[1], reverse=True)

            if self.gpu_support == "cuda":
                torch.cuda.empty_cache()
            ranked_scores = self._HappyTransformer__format_option_scores(ranked_scores)
            return ranked_scores

    def soft_sum(self, option: list, softed, mask_id: int):
        # TODO: Better logic.
        """
        Adds the softmax of a single option
        XLNET tokenizer sometimes splits words in to pieces.
        Ex: The councilmen -> ['the', 'council', 'men']
        Pretty sure that this is mathematically wrong
        :param option: Id of tokens in one option
        :param softed: softmax of the output
        :param mask: Index of masked word
        :return: float Tensor
        """
        # Collects the softmax of all tokens in list
        options = [softed[mask_id][op] for op in option]
        return np.sum(options)
