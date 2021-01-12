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
    assert result.loss == 0.11738169193267822


def test_qa_test():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.test("../data/qa/test.csv")
    answer = [QuestionAnsweringResult(answer='October 31st', score=0.9939756989479065, start=0, end=12),
              QuestionAnsweringResult(answer='November 23rd', score=0.967872679233551, start=12, end=25)]
    assert result == answer


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


def test_qa_train_effectiveness_albert():
    """
    Ensures that HappyQuestionAnswering.train() results in
    lowering the loss as determined by HappyQuestionAnswering.eval()
    """

    happy_qa = HappyQuestionAnswering("ALBERT", "twmkn9/albert-base-v2-squad2")
    before_loss = happy_qa.eval("../data/qa/train-eval.csv").loss
    happy_qa.train("../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../data/qa/train-eval.csv").loss

    assert after_loss < before_loss


def test_qa_test_albert():
    happy_qa = HappyQuestionAnswering("ALBERT", "twmkn9/albert-base-v2-squad2")
    result = happy_qa.test("../data/qa/test.csv")
    answer = [QuestionAnsweringResult(answer='October 31st', score=0.988578736782074, start=0, end=12),
              QuestionAnsweringResult(answer='November 23rd', score=0.9833534359931946, start=12, end=25)]
    assert result == answer


def test_qa_train_effectiveness_bert():
    """
    Ensures that HappyQuestionAnswering.train() results in
    lowering the loss as determined by HappyQuestionAnswering.eval()
    """

    happy_qa = HappyQuestionAnswering("BERT", "mrm8488/bert-tiny-5-finetuned-squadv2")
    before_loss = happy_qa.eval("../data/qa/train-eval.csv").loss
    happy_qa.train("../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../data/qa/train-eval.csv").loss

    assert after_loss < before_loss


def test_qa_test_bert():
    happy_qa = HappyQuestionAnswering("BERT", "mrm8488/bert-tiny-5-finetuned-squadv2")
    result = happy_qa.test("../data/qa/test.csv")
    answer = [QuestionAnsweringResult(answer='October 31st', score=0.9352769255638123, start=0, end=12),
              QuestionAnsweringResult(answer='November 23rd', score=0.9180678129196167, start=12, end=25)]
    assert result == answer


def test_qa_train_effectiveness_roberta():
    """
    Ensures that HappyQuestionAnswering.train() results in
    lowering the loss as determined by HappyQuestionAnswering.eval()
    """

    happy_qa = HappyQuestionAnswering("ROBERTA", "deepset/roberta-base-squad2")
    before_loss = happy_qa.eval("../data/qa/train-eval.csv").eval_loss
    happy_qa.train("../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../data/qa/train-eval.csv").eval_loss
    assert after_loss < before_loss


def test_qa_test_roberta():
    happy_qa = HappyQuestionAnswering("ROBERTA", "deepset/roberta-base-squad2")
    result = happy_qa.test("../data/qa/test.csv")
    answer = [QuestionAnsweringResult(answer='October 31st', score=0.9512737393379211, start=0, end=12),
              QuestionAnsweringResult(answer='November 23rd', score=0.8634917736053467, start=12, end=25)]
    assert result == answer
