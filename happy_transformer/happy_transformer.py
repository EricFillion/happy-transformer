# disable pylint TODO warning
# pylint: disable=W0511


"""
HappyTransformer is a wrapper over pytorch_transformers to make it
easier to use.

"""
import string
import re
import torch
import numpy as np

#Tesing github
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

    def __init__(self):
        # Transformer and tokenizer set in child class
        self.transformer = None
        self.tokenizer = None
        # Child class sets to indicate which model is being used
        self.model = ''

        # GPU support
        self.gpu_support = torch.device("cuda" if torch.cuda.is_available()
                                        else "cpu")
        print("Using model:", self.gpu_support)

    def predict_mask(self, text: str):
        # child classes must implement
        pass

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
        print("Error, {} not found in the input".format(self.masked_token))
        return None # TODO: medium: find a proper way to deal with errors

    def __get_formatted_text(self, text):
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
        new_text.append(self.cls_token)

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
                    new_text.append(self.sep_token)
                # must be a middle punctuation

        new_text.append(self.sep_token)
        text = " ".join(new_text)

        return text

    def __get_segment_ids(self, tokenized_text: list):
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
        split_location = tokenized_text.index(self.sep_token)
        segment_ids = list()
        for i in range(0, len(tokenized_text)):
            if i <= split_location:
                segment_ids.append(0)
            else:
                segment_ids.append(1)
            # add exception case for XLNet
        return segment_ids

<<<<<<< HEAD
    def text_verification(self, text:str):

        # TODO return True if the text is okay, else return False
        # TODO: create a happy_transformer test class and create a test_verification method
        #       Include at least 3 test cases.
        # TODO: As said in happy_bert, put the print statements within here
        # TODO: Add text_verification to happy_xlnet  and happy_roberta. Be sure to pull the master branch
        #       Before making any changes to them.
        # TODO,  Add cases for the other masked tokens used in common transformer models



        if '[MASK]' not in text:
            return -1, 0
        elif '[CLS]' in text:
            index = s2.find('[CLS]')
            return -2, index
        elif '[SEP]' in text:
            index = s2.find('[SEP]')
            return -3, index
        elif '##eer' in text: # TODO, dont need this check
            index = s2.find('[##eer]')
            return -4, index


=======
    def finish_sentence(self, text: str, maxPredictionLength = 100):
        """

        :param text: a string that is the start of a sentence to be finished
        :param maxPredictionLength: an int with the maximum number of words to be predicted
        :return: the completed sentence
        """
        father_predict = ""
        grand_father_predict = ""

        for i in range(0, maxPredictionLength):
            predict_text = text + self.masked_token
            predict_word = self.predict_mask(predict_text)[0]

            if predict_word == father_predict and predict_word == grand_father_predict:
                # if the same token was predicted three times in a row
                return text

            grand_father_predict = father_predict
            father_predict = predict_word

            text = text + predict_word
        return text

    def __get_tensors_and_mask_idx(self, text):
        formatted_text = self.__get_formatted_text(text)
        tokenized_text = self.tokenizer.tokenize(formatted_text)

        masked_index = self.__get_prediction_index(tokenized_text)
        segments_ids = self.__get_segment_ids(tokenized_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.gpu_support)
        segments_tensors = torch.tensor([segments_ids]).to(self.gpu_support)
        return tokens_tensor, segments_tensors, masked_index

    def __format_option_scores(self, ranked_scores: list):
        """
        :param: ranked_scores: list of tuples to be converted into user friendly dicitonary
        :return: formatted_ranked_scores: list of dictionaries of the ranked scores
        """
        # TODO: Shouldn't depend on ranked_scores to already be in order
        formatted_ranked_scores = list()
        for word, score in ranked_scores:
            formatted_ranked_scores.append({'word': word, 'score': score})
        return formatted_ranked_scores

    @staticmethod
    def soft_sum(option: list, softed, mask_id: int):
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
>>>>>>> master
