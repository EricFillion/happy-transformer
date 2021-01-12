"""
Tests for the question answering training, evaluating and testing functionality
"""

from happytransformer.happy_question_answering import HappyQuestionAnswering, QuestionAnsweringResult
from pytest import approx

def test_qa_answer_question():
    happy_qa = HappyQuestionAnswering()
    answers = happy_qa.answer_question("Today's date is January 8th 2021", "What is the date?")
    top_answer = answers[0]
    assert top_answer.answer == 'January 8th 2021'
    assert top_answer.start == 16
    assert top_answer.end == 32

def test_qa_answer_question_top_k():
    happy_qa = HappyQuestionAnswering()
    answers = happy_qa.answer_question("Today's date is January 8th 2021", "What is the date?", top_k=3)

    assert sum(answer.score for answer in answers) == approx(1,0.01)
    assert answers[0].start==16 and answers[0].end==32 and answers[0].answer=='January 8th 2021'
    assert answers[1].start==16 and answers[1].end==27 and answers[1].answer=='January 8th'


def test_qa_train():
    happy_qa = HappyQuestionAnswering()
    happy_qa.train("../data/qa/train-eval.csv")


def test_qa_eval():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.eval("../data/qa/train-eval.csv")
    assert result.loss == approx(0.11738169193267822,0.001)

def test_qa_test():
    happy_qa = HappyQuestionAnswering()
    results = happy_qa.test("../data/qa/test.csv")
    assert results[0].answer == 'October 31st'
    assert results[1].answer == 'November 23rd'

def test_qa_train_effectiveness():
    """
    Ensures that HappyQuestionAnswering.train() results in
    lowering the loss as determined by HappyQuestionAnswering.eval()
    """

    happy_qa = HappyQuestionAnswering()
    before_loss = happy_qa.eval("../data/qa/train-eval.csv").loss
    happy_qa.train("../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../data/qa/train-eval.csv").loss

    assert after_loss < before_loss
