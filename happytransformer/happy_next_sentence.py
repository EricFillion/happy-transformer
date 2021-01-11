import torch
import re
from transformers import (
    BertTokenizerFast,
    BertForNextSentencePrediction,

)

from happytransformer.happy_transformer import HappyTransformer


class HappyNextSentence(HappyTransformer):
    """
    A user facing class for next sentence prediction
    """
    def __init__(self, model_type="BERT",
                 model_name="bert-base-uncased"):

        if model_type == "BERT":
            model = BertForNextSentencePrediction.from_pretrained(model_name)
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
        else:
            raise ValueError(self.model_type_error)
        super().__init__(model_type, model_name, model, tokenizer)
        self._pipeline = None
        self._trainer = None

    def predict_next_sentence(self, sentence_a, sentence_b):
        """
        Determines if sentence B is likely to be a continuation after sentence
        A.
        :param sentence_a (string): First sentence
        :param sentence_b (string): Second sentence to test if it comes after the first
        :return (float): The probability that sentence_b follows sentence_a
        """
        if not self.__is_one_sentence(sentence_a) or not self.__is_one_sentence(sentence_b):
            raise ValueError('Each inputted text variable for the "predict_next_sentence" method must contain a single sentence')

        encoded = self._tokenizer(sentence_a, sentence_b, return_tensors='pt')
        with torch.no_grad():
            scores = self._model(encoded['input_ids'], token_type_ids=encoded['token_type_ids']).logits[0]

        probabilities = torch.softmax(scores, dim=0)
        # probability that sentence B follows sentence A
        score = probabilities[0].item()

        if self._device == 'cuda':
            torch.cuda.empty_cache()

        return score

    def __is_one_sentence(self, text):
        """
        Used to verify the proper input requirements for sentence_relation.
        The text must contain no more than a single sentence.
        Casual use of punctuation is accepted, such as using multiple exclamation marks.
        :param text: A body of text
        :return: True if the body of text contains a single sentence, else False
        """
        split_text = re.split('[?.!]', text)
        sentence_found = False
        for possible_sentence in split_text:
            for char in possible_sentence:
                if char.isalpha():
                    if sentence_found:
                        return False
                    sentence_found = True
                    break
        return True

    def train(self, input_filepath, args):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath):
        raise NotImplementedError("test() is currently not available")
