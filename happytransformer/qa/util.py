from collections import namedtuple
import torch
from happytransformer.tasks_util import biggest_sums

QAAnswerLogit = namedtuple('QaAnswerLogit', [
    'start_idx', 'end_idx', 'logit'
])

def qa_logits(start_logits, end_logits):
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
        QAAnswerLogit(
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

def qa_probabilities(start_logits, end_logits, k):
    """
    Computes the top k qa probabilities, in terms of indices.
    :param start_logits: tensor from qa model output
    :param end_logits: tensor from qa model output
    :param k: number of results to return
    :returns: list of namedtuples of the form (text,probability)
    """
    top_answers = [
        qa_logit
        for qa_logit, _ in zip(qa_logits(start_logits, end_logits), range(k))
    ]
    logit_scores = torch.tensor([
        answer.logit
        for answer in top_answers
    ])

    probabilities = torch.nn.Softmax(dim=0)(logit_scores).tolist()
    return [
        QAProbability(
            start_idx=answer.start_idx,
            end_idx=answer.end_idx,
            probability=probability
        )
        for answer, probability in zip(top_answers, probabilities)
    ]
