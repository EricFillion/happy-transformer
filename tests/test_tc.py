"""
Tests for Text Classification Functionality
"""

from happytransformer import(
    HappyTextClassification,
    TCTrainArgs,
    TCEvalArgs,
    TCTestArgs
)
from tests.run_save_load import run_save_load
from pytest import approx
import pytest

def test_classify_text():
    happy_tc = HappyTextClassification(model_type="DISTILBERT", model_name="distilbert-base-uncased-finetuned-sst-2-english")
    result = happy_tc.classify_text("What a great movie")
    assert result.label == 'POSITIVE'
    assert result.score > 0.9


def test_tc_train():
    happy_tc = HappyTextClassification(
        model_type="BERT",
        model_name="prajjwal1/bert-tiny"
    )
    results = happy_tc.train("../data/tc/train-eval.csv")


def test_tc_eval():
    happy_tc = HappyTextClassification(
        model_type="DISTILBERT",
        model_name="distilbert-base-uncased-finetuned-sst-2-english"
    )
    results = happy_tc.eval("../data/tc/train-eval.csv")
    assert results.loss == approx(0.007262040860950947, 0.01)


def test_tc_test():
    happy_tc = HappyTextClassification(
        model_type="DISTILBERT",
        model_name="distilbert-base-uncased-finetuned-sst-2-english"
    )

    result = happy_tc.test("../data/tc/test.csv")

    labels_result = [case.label for case in result]
    answer = [
        'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE'
    ]
    assert labels_result == answer


def test_tc_train_effectiveness():
    """assert that training decreases the loss"""
    happy_tc = HappyTextClassification(
        model_type="BERT",
        model_name="prajjwal1/bert-tiny"
    )
    before_loss = happy_tc.eval("../data/tc/train-eval.csv").loss
    happy_tc.train("../data/tc/train-eval.csv")
    after_loss = happy_tc.eval("../data/tc/train-eval.csv").loss
    assert after_loss < before_loss

#todo investigate why this passes locally but fails on GitHub actions

def test_tc_train_effectiveness_multi():
    
    happy_tc = HappyTextClassification(
        model_type="BERT",
        model_name="prajjwal1/bert-tiny",
        num_labels=3 )

    before_loss = happy_tc.eval("../data/tc/train-eval-multi.csv").loss
    happy_tc.train("../data/tc/train-eval-multi.csv")
    after_loss = happy_tc.eval("../data/tc/train-eval-multi.csv").loss
    assert after_loss < before_loss



#TODO investigate why with some models the labels change after is saved and loaded
def test_tc_save():
    happy = HappyTextClassification(model_type="DISTILBERT",
        model_name="distilbert-base-uncased", num_labels=3)
    happy.save("model/")
    result_before = happy.classify_text("What a great movie")

    happy = HappyTextClassification(model_name="model/", num_labels=3)
    result_after = happy.classify_text("What a great movie")

    assert result_before.label==result_after.label
    

def test_tc_with_dic():

    happy_tc = HappyTextClassification(model_type="BERT",
        model_name="prajjwal1/bert-tiny")
    train_args = {'learning_rate': 0.01,  "num_train_epochs": 1}

    with pytest.raises(ValueError):
        happy_tc.train("../data/tc/train-eval.csv" , args=train_args)

    eval_args = {}
    with pytest.raises(ValueError):
        result_eval = happy_tc.eval("../data/tc/train-eval.csv", args=eval_args)

    test_args = {}
    with pytest.raises(ValueError):
        result_test = happy_tc.test("../data/tc/test.csv", args=test_args)


def test_tc_with_dataclass():

    happy_tc = HappyTextClassification(model_type="BERT",
        model_name="prajjwal1/bert-tiny")
    train_args = TCTrainArgs(learning_rate=0.01, num_train_epochs=1)

    happy_tc.train("../data/tc/train-eval.csv", args=train_args)

    eval_args = TCEvalArgs()

    result_eval= happy_tc.eval("../data/tc/train-eval.csv", args=eval_args)


    test_args = TCTestArgs()

    result_test = happy_tc.test("../data/tc/test.csv", args=test_args)

def test_tc_save_load_train():
    happy_wp = HappyTextClassification(model_type="BERT",
        model_name="prajjwal1/bert-tiny")

    output_path = "data/tc-train/"
    data_path = "../data/tc/train-eval.csv"
    args = TCTrainArgs(num_train_epochs=1)
    run_save_load(happy_wp, output_path, args, data_path, "train")


def test_tc_save_load_eval():
    happy_wp = HappyTextClassification(model_type="DISTILBERT", model_name="distilbert-base-uncased-finetuned-sst-2-english")
    output_path = "data/tc-eval/"
    data_path = "../data/tc/train-eval.csv"
    args = TCEvalArgs()

    run_save_load(happy_wp, output_path, args, data_path, "eval")
