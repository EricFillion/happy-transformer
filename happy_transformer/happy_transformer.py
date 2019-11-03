# disable pylint TODO warning
# pylint: disable=W0511


"""
HappyTransformer is a wrapper over pytorch_transformers to make it
easier to use.

"""
import string
import re
import torch


# TODO create a test framework
# TODO: easy: find other challenges we can test the masked word prediction on

class HappyTransformer:
    """
    Initializes pytroch's transformer models and provided methods for
    their basic functionality.

    Philosophy: Automatically make decisions for the user so that they don't have to
                have any understanding of PyTorch or transformer models to be able
                to utilize their capabilities.
    """

    # TODO: complex: Turn this class into a parent class and then create child classes for it.
    #       Some child classes would include  HappyBERT, HappyXLNet e

    # TODO: complex: think of a unique public method for HappyBERT and implement it
    #                eg, sentence perplexity and rank words

    # TODO: 10/10 hard: fine tuning module

    def __init__(self):
        # Transformer and tokenizer set in child class
        self.transformer = None
        self.tokenizer = None

        # GPU support
        self.gpu_support = torch.device("cuda" if torch.cuda.is_available()
                                        else "cpu")
        print("Using model:", self.gpu_support)

    def predict_mask(self, text: str):
        """
        :param text: a string with a masked token within it
        :return: predicts the most likely word to fill the mask and its probability
        """

        # TODO: put in HappyBERT. Overwrite HappyTransformer.
        # TODO: easy: create a method to check if the sentence is valid
        # TODO: easy: if the sentence is not valid, provide the user with input requirements
        # TODO: easy: if sentence is not valid, indicate where the user messed up

        # TODO: medium: make it more modular

        # Generated formatted text so that it can be tokenized. Mainly, add the required tags
        formatted_text = self.__get_formatted_text(text)
        tokenized_text = self.tokenizer.tokenize(formatted_text)

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

            softmax = self.__softmax(predictions)

            top_prediction = torch.topk(softmax[0, masked_index], 1)
            prediction_softmax = top_prediction[0].tolist()
            prediction_index = top_prediction[1].tolist()

            prediction_token = self.tokenizer.convert_ids_to_tokens(prediction_index)

           # TODO: easy: del various variables

            if self.gpu_support == "cuda":
                torch.cuda.empty_cache()

            return prediction_token, prediction_softmax

    def __softmax(self, value):
        # TODO: make it an external function
        return value.exp() / (value.exp().sum(-1)).unsqueeze(-1)

    def __get_prediction_index(self, tokenized_text):
        """
        Gets the location of the first occurrence of the [MASK] index
        :param tokenized_text: a list of word tokens where one of the tokens is the string "[MASK]"
        :return:
        """
        # TODO: put in HappyBERT. Overwrite HappyTransformer.
        #  Maybe only the masked token needs to be changed per HappyClass

        # TODO: easy: there might be a cleaner way to do this
        location = 0
        for token in tokenized_text:
            if token == self.masked_token:
                return location
            location += 1
        print("Error, {} not found in the input".format(self.mask_token))
        return None # TODO: medium: find a proper way to deal with errors

    def __get_formatted_text(self, text):
        # TODO: put in HappyBERT. Overwrite HappyTransformer
        """
        Formats a sentence so that BERT it can be tokenized by BERT.

        :param text: a 1-2 sentence text that contains [MASK]
        :return: A string with the same sentence that contains the required tokens for BERT
        """

        # Create a spacing around each punctuation character. eg "!" -> " ! "
        # TODO: easy: find a cleaner way to do punctuation spacing
        text = re.sub('([.,!?()])', r' \1 ', text)
        # text = re.sub('\s{2,}', ' ', text)

        split_text = text.split()
        new_text = list()
        new_text.append(self.classification_token)

        for i, char in enumerate(split_text):
            new_text.append(char)
            if char not in string.punctuation:
                pass
            # must be a punctuation symbol
            elif i+1 >= len(split_text):
                # is the last punctuation so simply add to the new_text
                pass
            else:
                if split_text[i + 1] in string.punctuation:
                    pass
                else:
                    new_text.append(self.separate_token)
                # must be a middle punctuation

        new_text.append(self.separate_token)
        text = " ".join(new_text)

        return text

    def __get_segment_ids(self, tokenized_text: list):
        # TODO: put in HappyBERT
        """
        Converts a list of tokens into segment_ids. The segment id is a array
        representation of the location for each character in the
        first and second sentence. This method only words with 1-2 sentences.

        Example:
        tokenized_text = ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]',
                         'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        returns segments_ids
        """
        split_location = tokenized_text.index(self.separate_token)
        segment_ids = list()
        for i in range(0, len(tokenized_text)):
            if i <= split_location:
                segment_ids.append(0)
            else:
                segment_ids.append(1)
        return segment_ids
