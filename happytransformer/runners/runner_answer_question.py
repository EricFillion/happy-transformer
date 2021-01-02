import torch

from happytransformer.runners.runner_util import biggest_sums
from collections import namedtuple

class QuestionAnswering():
    def __init__(self, type, model, tokenizer):
        self.type = type # BERT, ROBERTA, ALBERT etc
        self.model = model
        self.tokenizer = tokenizer

    def answers_to_question(self,  question, context, k=3):
        input_ids = self._tokenize_qa(question, context)
        qa_output = self._run_qa_model(input_ids)
        sep_id_index = input_ids.index(self.tokenizer.sep_token_id)
        probabilities = self.qa_probabilities(
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

    def answer_question(self, question, text):
        """
        Using the given text, find the answer to the given question and return it.

        :param question: The question to be answered
        :param text: The text containing the answer to the question
        :return: The answer to the given question, as a string
        """
        return self.answers_to_question(question, text,  1)[0]["text"]

    def _tokenize_qa(self, question, context):
        input_text = ' '.join([
            question,
            self.tokenizer.sep_token,
            context
        ])
        input_ids = self.tokenizer.encode(input_text)
        return input_ids


    def _run_qa_model(self, input_ids):

        sep_id_index = input_ids.index(self.tokenizer.sep_token_id)
        before_after_ids = [
            0 if idx <= sep_id_index else 1
            for idx, _ in enumerate(input_ids)
        ]
        with torch.no_grad():
            return self.model(
                input_ids=torch.tensor([input_ids]),
                token_type_ids=torch.tensor([before_after_ids])
            )

    QAAnswerLogit = namedtuple('QaAnswerLogit', [
        'start_idx', 'end_idx', 'logit'
    ])

    def qa_logits(self, start_logits, end_logits):
        """
        Compute the logits for top qa pairs
        :param start_logits: tensor from qa model output
        :param end_logits: tensor from qa model output
        :returns: generator of namedtuples of the form
        (start_idx, end_idx, logit), sorted in descending order
        by score
        """

        sorted_starts_tensors = torch.sort(start_logits, descending=True)
        sorted_ends_tensors = torch.sort(end_logits, descending=True)
        # start logits sorted in descending order INDEPENDENTLY
        sorted_start_scores = sorted_starts_tensors.values.tolist()
        sorted_start_indices = sorted_starts_tensors.indices.tolist()
        # end logits sorted in descending order INDEPENDENTLY
        sorted_end_scores = sorted_ends_tensors.values.tolist()
        sorted_end_indices = sorted_ends_tensors.indices.tolist()
        # start logit + end logit pairs sorted in descending order
        # of their sum TOGETHER
        all_answers = (
            self.QAAnswerLogit(
                start_idx=sorted_start_indices[sum_pair.idx1],
                end_idx=sorted_end_indices[sum_pair.idx2],
                logit=sum_pair.sum
            )
            for sum_pair in
            biggest_sums(sorted_start_scores, sorted_end_scores)
        )
        # filter for only answers which have end at or after start
        legit_answers = (
            answer
            for answer in all_answers
            if answer.end_idx >= answer.start_idx
        )
        return legit_answers

    QAProbability = namedtuple('QaProbability', [
        'start_idx', 'end_idx', 'probability'
    ])

    def qa_probabilities(self, start_logits, end_logits, k):
        """
        Computes the top k qa probabilities, in terms of indices.
        :param start_logits: tensor from qa model output
        :param end_logits: tensor from qa model output
        :param k: number of results to return
        :returns: list of namedtuples of the form (text,probability)
        """
        top_answers = [
            qa_logit
            for qa_logit, _ in zip(self.qa_logits(start_logits, end_logits), range(k))
        ]
        logit_scores = torch.tensor([
            answer.logit
            for answer in top_answers
        ])

        probabilities = torch.nn.Softmax(dim=0)(logit_scores).tolist()
        return [
            self.QAProbability(
                start_idx=answer.start_idx,
                end_idx=answer.end_idx,
                probability=probability
            )
            for answer, probability in zip(top_answers, probabilities)
        ]
