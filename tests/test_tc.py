"""
Tests for Text Classification Functionality
"""

from happytransformer.happy_text_classification import HappyTextClassification, TextClassificationResult
from pytest import approx


def test_classify_text():
    MODELS = [
        ('DISTILBERT', 'distilbert-base-uncased-finetuned-sst-2-english'),
        ("ALBERT", "textattack/albert-base-v2-SST-2")
    ]
    for model_type, model_name in MODELS:
        happy_tc = HappyTextClassification(model_type=model_type, model_name=model_name)
        result = happy_tc.classify_text("What a great movie")
        assert result.label == 'LABEL_1'
        assert result.score > 0.9


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
    answer = [
        TextClassificationResult(label='LABEL_1', score=0.9998401999473572),
        TextClassificationResult(label='LABEL_0', score=0.9772131443023682),
        TextClassificationResult(label='LABEL_0', score=0.9966067671775818),
        TextClassificationResult(label='LABEL_1', score=0.9792295098304749)
    ]
    assert result == answer


def test_tc_train_effectiveness():
    """assert that training decreases the loss"""
    happy_tc = HappyTextClassification(
        model_type="DISTILBERT",
        model_name="distilbert-base-uncased"
    )
    before_loss = happy_tc.eval("../data/tc/train-eval.csv").loss
    happy_tc.train("../data/tc/train-eval.csv")
    after_loss = happy_tc.eval("../data/tc/train-eval.csv").loss
    assert after_loss < before_loss


def test_tc_train_effectiveness_multi():
    
    happy_tc = HappyTextClassification(
        model_type="DISTILBERT",
        model_name="distilbert-base-uncased", 
        num_labels=3
    )
    before_loss = happy_tc.eval("../data/tc/train-eval-multi.csv").loss
    happy_tc.train("../data/tc/train-eval-multi.csv")
    after_loss = happy_tc.eval("../data/tc/train-eval-multi.csv").loss
    assert after_loss < before_loss
