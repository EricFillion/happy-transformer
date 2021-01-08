"""
Tests for Text Classification Functionality
"""

from happytransformer.happy_text_classification import HappyTextClassification

def test_classify_text():
    """
    Tests
    HappyQuestionAnswering.classify_text()

    """
    happy_tc = HappyTextClassification()
    result = happy_tc.classify_text("What a great movie")
    assert result["answer"] == 1 and result["softmax"][1] == 0.9998726844787598

def test_qa_train():
    """
    Tests
    HappyQuestionAnswering.train()

    """
    happy_tc = HappyTextClassification()

    happy_tc.train("../data/tc/train-eval.csv")


def test_qa_eval():
    """
    Tests
    HappyQuestionAnswering.eval()
    """
    happy_tc = HappyTextClassification()
    results = happy_tc.eval("../data/tc/train-eval.csv")
    assert results["eval_loss"] == 0.007262040860950947


def test_qa_test():
    """
    Tests
    HappyQuestionAnswering.test()
    """
    happy_tc = HappyTextClassification()

    result = happy_tc.test("../data/tc/test.csv")
    expected_result = [[0.00015978473364387978, 0.9998402152663561], [0.9772132247336673, 0.022786775266332746], [0.9966067733093962, 0.0033932266906038368], [0.020770484301764973, 0.979229515698235]]

    assert result == expected_result


def test_qa_train_effectiveness():
    """
    Tests
    Ensures that HappyQuestionAnswering.train() results in
    lowering the loss as determined by HappyQuestionAnswering.eval()
    """

    happy_tc = HappyTextClassification()
    before_loss = happy_tc.eval("../data/tc/train-eval.csv")["eval_loss"]
    happy_tc.train("../data/tc/train-eval.csv")
    after_loss = happy_tc.eval("../data/tc/train-eval.csv")["eval_loss"]
    assert after_loss < before_loss
