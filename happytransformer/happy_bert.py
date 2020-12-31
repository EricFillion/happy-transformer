"""
HappyBERT: a wrapper over PyTorch's BERT implementation

"""

from collections import namedtuple
# disable pylint TODO warning
# pylint: disable=W0511
import re
from transformers import (
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForQuestionAnswering,
    BertTokenizerFast
)

import torch
import numpy as np

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.qa_util import qa_probabilities

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
        self.tokenizer = BertTokenizerFast.from_pretrained(model)

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

    def answer_question(self, question, text):
        """
        Using the given text, find the answer to the given question and return it.

        :param question: The question to be answered
        :param text: The text containing the answer to the question
        :return: The answer to the given question, as a string
        """
        return self.answers_to_question(question, text, 1)[0]["text"]

    def _tokenize_qa(self, question, context):
        input_text = ' '.join([
            question, 
            self.tokenizer.sep_token,
            context
        ])
        input_ids = self.tokenizer.encode(input_text)
        return input_ids

    def _run_qa_model(self, input_ids):
        if self.qa is None:
            self._get_question_answering()
        sep_id_index = input_ids.index(self.tokenizer.sep_token_id)
        before_after_ids = [
            0 if idx <= sep_id_index else 1
            for idx, _ in enumerate(input_ids)
        ]
        with torch.no_grad():
            return self.qa(
                input_ids=torch.tensor([input_ids]),
                token_type_ids=torch.tensor([before_after_ids])
            )

    def answers_to_question(self, question, context, k=3):
        input_ids = self._tokenize_qa(question, context)
        qa_output = self._run_qa_model(input_ids)
        sep_id_index = input_ids.index(self.tokenizer.sep_token_id)
        probabilities = qa_probabilities(
            # only consider logits from the context part of the embedding.
            # that is, between the middle [SEP] token
            # and the final [SEP] token
            qa_output.start_logits[0][sep_id_index+1:-1],
            qa_output.end_logits[0][sep_id_index+1:-1],
            k
        )
        # qa probabilities use indices relative to context.
        # tokens use indices relative to overall question [SEP] context embedding.
        # need offset to resolve this difference
        token_offset = sep_id_index + 1

        return [
            {"text": self.tokenizer.decode(
                    # grab ids from start to end (inclusive) and decode to text
                    input_ids[token_offset+answer.start_idx : token_offset+answer.end_idx+1]
                ),
            "softmax": answer.probability}

            for answer in probabilities
        ]
