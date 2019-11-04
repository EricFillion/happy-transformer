
import torch
from transformers import XLNetLMHeadModel,XLNetTokenizer

from happy_transformer import HappyTransformer

class HappyXLNET(HappyTransformer):
    """

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

        # formatted_text = self._HappyTransformer__get_formatted_text(text) cant get it to word
        tokenized_text = self.tokenizer.tokenize(text)

        masked_index = self.__get_prediction_index(tokenized_text)
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
            del outputs, softmax,predictions, top_prediction, prediction_index

            if self.gpu_support == "cuda":
                torch.cuda.empty_cache()

            return prediction_token, prediction_softmax

    def predict_mask_with_options(self, text: str, options: list):
        """

        :param text: a string with a masked token within it
        :param options: a list of strings as options for masked word
        :return: predicts the most likely word from list of options to fill the mask and its probability
        """

        # formatted_text = self._HappyTransformer__get_formatted_text(text) Not sure if needed
        tokenized_text = self.tokenizer.tokenize(text)

        masked_index = self.__get_prediction_index(tokenized_text)
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

            softmax = self._HappyTransformer__softmax(predictions)[0]

            option_ids = list()
            for option in options:
                option_ids.append(self.tokenizer.encode(option))

            option_probs = list(map(lambda x: self.soft_sum(x, softmax, masked_index), option_ids))
            tupled_option = tuple(zip(options, option_probs))
            prediction_token = sorted(tupled_option, key=lambda x: x[1], reverse=True)

            if self.gpu_support == "cuda":
                torch.cuda.empty_cache()

            return prediction_token

    def __get_segment_ids(self, tokenized_text: list): # Temp fix
        return [0] * len(tokenized_text)


    def __get_prediction_index(self, tokenized_text):
        """
        Gets the location of the first occurrence of the [MASK] index
        :param tokenized_text: a list of word tokens where one of the tokens is the string "[MASK]"
        :return:
        """
        # TODO: put in HappyBERT. Overwrite HappyTransformer.
        #  Maybe only the masked token needs to be changed per HappyClass

        # TODO: easy: there might be a cleaner way to do this

        return tokenized_text.index('<mask>')



    def soft_sum(self, option, softed, mask):
        # TODO: Better logic.
        """

        Pretty sure that this is mathematically wrong(Can't add dependent probabilities)

        :param option:
        :param softed:
        :param mask:
        :return: float Tensor
        """
        options = []
        for op in option:
            options.append(softed[mask][op])
        return sum(options)


def main():
    happy = HappyXLNET()
    print(happy.predict_mask_with_options("My dog is very <mask>",['cute','friendly','cat','crazy']))

if __name__ == '__main__':
    main()