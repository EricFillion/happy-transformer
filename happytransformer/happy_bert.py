"""
HappyBERT: a wrapper over PyTorch's BERT implementation

"""

# disable pylint TODO warning
# pylint: disable=W0511
import re
from transformers import (
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForQuestionAnswering,
    BertTokenizer
)

import torch

from happytransformer.happy_transformer import HappyTransformer


class HappyBERT(HappyTransformer):
    """
    Currently available public methods:
        BertForMaskedLM:
            1. predict_mask(text: str, options=None, k=1)
        BertForSequenceClassification:
            1. init_sequence_classifier()
            2. advanced_init_sequence_classifier()
            3. train_sequence_classifier(train_csv_path)
            4. eval_sequence_classifier(eval_csv_path)
            5. test_sequence_classifier(test_csv_path)
        BertForNextSentencePrediction:
            1. predict_next_sentence(sentence_a, sentence_b)
        BertForQuestionAnswering:
            1. answer_question(question, text)

            """

    def __init__(self, model='bert-base-uncased'):
        super().__init__(model, "BERT")
        self.mlm = None  # Masked Language Model
        self.nsp = None  # Next Sentence Prediction
        self.qa = None   # Question Answering
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token

    def _get_masked_language_model(self):
        """
        Initializes the BertForMaskedLM transformer
        """
        self.mlm = BertForMaskedLM.from_pretrained(self.model)
        self.mlm.eval()

    def _get_next_sentence_prediction(self):
        """
        Initializes the BertForNextSentencePrediction transformer
        """
        self.nsp = BertForNextSentencePrediction.from_pretrained(self.model)
        self.nsp.eval()

    def _get_question_answering(self):
        """
        Initializes the BertForQuestionAnswering transformer
        NOTE: This uses the bert-large-uncased-whole-word-masking-finetuned-squad pretraining for best results.
        """
        self.qa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.qa.eval()

    def predict_next_sentence(self, sentence_a, sentence_b, use_probability=False):
        """
        Determines if sentence B is likely to be a continuation after sentence
        A.
        :param sentence_a: First sentence
        :param sentence_b: Second sentence to test if it comes after the first
        :param use_probability: Toggle outputting probability instead of boolean
        :return Result of whether sentence B follows sentence A,
                as either a probability or a boolean
        """

        if not self.__is_one_sentence(sentence_a) or not  self.__is_one_sentence(sentence_b):
            self.logger.error("Each inputted text variable for the \"predict_next_sentence\" method must contain a single sentence")
            exit()

        if self.nsp is None:
            self._get_next_sentence_prediction()
        connected = sentence_a + ' ' + sentence_b
        tokenized_text = self._get_tokenized_text(connected)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = self._get_segment_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            predictions = self.nsp(tokens_tensor, token_type_ids=segments_tensors)[0]

        probabilities = torch.nn.Softmax(dim=1)(predictions)
        # probability that sentence B follows sentence A
        correct_probability = probabilities[0][0].item()

        return (
            correct_probability if use_probability else 
            correct_probability >= 0.5
        )

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

    def answer_question(self, question, text):
        """
        Using the given text, find the answer to the given question and return it.

        :param question: The question to be answered
        :param text: The text containing the answer to the question
        :return: The answer to the given question, as a string
        """
        if self.qa is None:
            self._get_question_answering()
        input_text = self.cls_token + " " + question + " " + self.sep_token + " " + text + " " + self.sep_token
        input_ids = self.tokenizer.encode(input_text)
        sep_val = self.tokenizer.encode(self.sep_token)[-1]
        token_type_ids = [0 if i <= input_ids.index(sep_val) else 1
                          for i in range(len(input_ids))]
        token_tensor = torch.tensor([input_ids])
        segment_tensor = torch.tensor([token_type_ids])
        with torch.no_grad():
            start_scores, end_scores = self.qa(input_ids=token_tensor,
                                               token_type_ids=segment_tensor)
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer_list = all_tokens[torch.argmax(start_scores):
                                 torch.argmax(end_scores)+1]
        answer = self.tokenizer.convert_tokens_to_string(answer_list)
        answer = answer.replace(' \' ', '\' ').replace('\' s ', '\'s ')
        return answer
