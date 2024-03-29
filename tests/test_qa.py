"""
Tests for the question answering training, evaluating and testing functionality
"""

from happytransformer.happy_question_answering import (
    HappyQuestionAnswering,
    QATrainArgs,
    QAEvalArgs,
    QATestArgs
)
import pytest
from pytest import approx
from tests import happy_qa

def test_qa_basic():
    result = happy_qa.answer_question("Today's date is January 8th 2021", "What is the date?", top_k=10)
    assert result[0].answer == 'January 8th 2021'
    assert result[0].score == approx(0.9696964621543884, 1)

    #assert sum(answer.score for answer in answers) == approx(1, 0.5)
    #assert all('January 8th' in answer.answer for answer in answers)


def test_qa_train():
    result = happy_qa.train("../data/qa/train-eval.csv")


def test_qa_eval():
    happy = HappyQuestionAnswering("DISTILBERT", "distilbert-base-cased-distilled-squad")
    result = happy.eval("../data/qa/train-eval.csv")
    assert result.loss == approx(0.11738169193267822, 0.01)


def test_qa_test():
    happy = HappyQuestionAnswering("DISTILBERT", "distilbert-base-cased-distilled-squad")

    results = happy.test("../data/qa/test.csv")
    assert results[0].answer == 'October 31st'
    assert results[1].answer == 'November 23rd'


def test_qa_train_effectiveness():
    """
    Ensures that HappyQuestionAnswering.train() results in
    lowering the loss as determined by HappyQuestionAnswering.eval()
    """
    # use a non-fine-tuned model so we DEFINITELY get an improvement
    happy = HappyQuestionAnswering("BERT", "bert-base-uncased")
    args = QATrainArgs(num_train_epochs=3)
    before_loss = happy.eval("../data/qa/train-eval.csv").loss
    happy.train("../data/qa/train-eval.csv", args=args)
    after_loss = happy.eval("../data/qa/train-eval.csv").loss

    assert after_loss < before_loss


def test_qa_save():

    happy_qa.save("model/")
    result_before = happy_qa.answer_question("Natural language processing is a subfield of artificial surrounding creating models that understand language","What is natural language processing?")

    happy = HappyQuestionAnswering(model_name="model/")
    result_after = happy.answer_question("Natural language processing is a subfield of artificial surrounding creating models that understand language","What is natural language processing?")

    assert result_before[0].answer == result_after[0].answer


def test_qa_with_dic():


    train_args = {'learning_rate': 0.01,  "num_train_epochs": 1}

    with pytest.raises(ValueError):
        happy_qa.train("../data/qa/train-eval.csv" , args=train_args)

    eval_args = {}

    with pytest.raises(ValueError):
        result_eval = happy_qa.eval("../data/qa/train-eval.csv", args=eval_args)

    test_args = {}
    with pytest.raises(ValueError):
        result_test = happy_qa.test("../data/qa/test.csv", args=test_args)

def test_tc_with_dataclass():


    train_args = QATrainArgs(learning_rate=0.01, num_train_epochs=1)

    happy_qa.train("../data/qa/train-eval.csv", args=train_args)

    eval_args = QAEvalArgs()

    result_eval = happy_qa.eval("../data/qa/train-eval.csv", args=eval_args)
    assert type(result_eval.loss) == float


    test_args = QATestArgs()

    result_test = happy_qa.test("../data/qa/test.csv", args=test_args)
    assert type(result_test[0].answer) == str
