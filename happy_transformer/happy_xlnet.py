# pylint: disable=W0511
import torch
import numpy as np
from transformers import XLNetLMHeadModel, XLNetTokenizer

from happy_transformer import HappyTransformer


class HappyXLNET(HappyTransformer):
    """
    Implementation of XLNET for masked word prediction
    """

    def __init__(self, model='xlnet-large-cased'):
        super().__init__()

        self.transformer = XLNetLMHeadModel.from_pretrained(model)
        self.tokenizer = XLNetTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer._mask_token
        self.sep_token = self.tokenizer._sep_token
        self.cls_token = self.tokenizer._cls_token

        self.model = 'XLNET'

        self.transformer.eval()

    def predict_mask(self, text: str):
        """
        :param text: a string with a masked token within it
        :return: predicts the most likely word to fill the mask and its probability
        """

        formatted_text = self._HappyTransformer__get_formatted_text(text)
        tokenized_text = self.tokenizer.tokenize(formatted_text)

        masked_index = self._HappyTransformer__get_prediction_index(tokenized_text)
        segments_ids = self.__get_segment_ids(tokenized_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        tokens_tensor = tokens_tensor.to(self.gpu_support)
        segments_tensors = segments_tensors.to(self.gpu_support)

        with torch.no_grad():
            outputs = self.transformer(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

            softmax = self._HappyTransformer__softmax(predictions)

            top_prediction = torch.topk(softmax[0, masked_index], 1)
            prediction_softmax = top_prediction[0].tolist()
            prediction_index = top_prediction[1].tolist()

            prediction_token = self.tokenizer.convert_ids_to_tokens(prediction_index)

            # TODO: easy: del various variables
            del outputs, softmax, predictions, top_prediction, prediction_index

            if self.gpu_support == "cuda":
                torch.cuda.empty_cache()

            return prediction_token, prediction_softmax

    def predict_mask_with_options(self, text: str, options: list):
        """

        :param text: a string with a masked token within it
        :param options: a list of strings as options for masked word
        :return: predicts the most likely word from list of options
        to fill the mask and its probability
        """

        formatted_text = self._HappyTransformer__get_formatted_text(text)
        tokenized_text = self.tokenizer.tokenize(formatted_text)

        masked_index = self._HappyTransformer__get_prediction_index(tokenized_text)
        segments_ids = self._HappyTransformer__get_segment_ids(tokenized_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        tokens_tensor = tokens_tensor.to(self.gpu_support)
        segments_tensors = segments_tensors.to(self.gpu_support)

        with torch.no_grad():
            outputs = self.transformer(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

            softmax = self._HappyTransformer__softmax(predictions)[0]

            option_ids = [self.tokenizer.encode(option) for option in options]

            option_probs = list(map(lambda x: self.soft_sum(x, softmax, masked_index), option_ids))
            tupled_option = tuple(zip(options, option_probs))
            prediction_token = sorted(tupled_option, key=lambda x: x[1], reverse=True)

            if self.gpu_support == "cuda":
                torch.cuda.empty_cache()

            return prediction_token

    def soft_sum(self, option: list, softed, mask_id: int):
        # TODO: Better logic.
        """
        Adds the softmax of a single option
        XLNET tokenizer sometimes splits words in to pieces.
        Ex: The councilmen -> ['the', 'council', 'men']
        Pretty sure that this is mathematically wrong


        :param option: Id of tokens in one option
        :param softed: softmax of the output
        :param mask_id: Index of masked word
        :return: float Tensor
        """
        # Collects the softmax of all tokens in list
        options = [softed[mask_id][op] for op in option]
        return np.sum(options)

