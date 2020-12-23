from collections import namedtuple
import torch

SumPair = namedtuple('SumPair',['idx1', 'idx2', 'sum'])

def biggest_sums(items_a, items_b):
    '''
    compute biggest sums from two descending ordered lists,
    labeled by indices
    :param items_a: list of numeric values, sorted descendingly
    :param items_b: list of numeric values, sorted descendingly
    :returns: list of namedtuples of the form (idx1,idx2,sum),
    sorted by descending sum
    '''
    a_index = b_index = 0
    while a_index < len(items_a) and b_index < len(items_b):
        yield SumPair(
            a_index, b_index,
            sum=items_a[a_index] + items_b[b_index]
        )
        # increment in whichever direction has smaller gain
        # fallback to -inf at end of list. 
        # this will always be taken last.
        next_from_a = items_a[a_index+1] if a_index + 1 < len(items_a) else float('-inf')
        next_from_b = items_b[b_index+1] if b_index + 1 < len(items_b) else float('-inf')

        diff_a = items_a[a_index] - next_from_a
        diff_b = items_b[b_index] - next_from_b

        if diff_a >= diff_b:
            b_index += 1
        else:
            a_index += 1

QaAnswerLogit = namedtuple('QaAnswerLogit', [
    'start_idx','end_idx', 'logit'
])

def qa_logits(start_logits, end_logits):
    '''
    Compute the logits for top qa pairs
    :param start_logits: tensor from qa model output
    :param end_logits: tensor from qa model output
    :returns: generator of namedtuples of the form
    (start_idx, end_idx, logit), sorted in descending order
    by score
    ''' 
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
        QaAnswerLogit(
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

QaProbability = namedtuple('QaProbability', [
    'start_idx', 'end_idx', 'probability'
])
QaAnswer = namedtuple('QaAnswer',[
    'text', 'probability'
])

def qa_probabilities(start_logits, end_logits, k):
    '''
    Computes the top k qa probabilities, in terms of indices.
    :param start_logits: tensor from qa model output
    :param end_logits: tensor from qa model output
    :param k: number of results to return
    :returns: list of namedtuples of the form (text,probability)
    '''
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
        QaProbability(
            start_idx=answer.start_idx,
            end_idx=answer.end_idx,
            probability=probability
        )
        for answer, probability in zip(top_answers, probabilities)
    ]