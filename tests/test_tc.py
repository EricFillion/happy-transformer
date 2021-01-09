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
    print(result)
    answer = [{'label': 'POSITIVE', 'score': 0.9998726844787598}]
    assert result == answer

def test_classify_texts():
    """
    Tests
    HappyQuestionAnswering.classify_text()

    """
    happy_tc = HappyTextClassification()
    input = ["What a great movie", "Horrible movie", "Bad restaurant"]
    result = happy_tc.classify_text(input)
    answer = [{'label': 'POSITIVE', 'score': 0.9998726844787598}, {'label': 'NEGATIVE', 'score': 0.9997945427894592}, {'label': 'NEGATIVE', 'score': 0.9997393488883972}]
    assert result == answer

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
    answer = [[{'label': 'POSITIVE', 'score': 0.9998401999473572}], [{'label': 'NEGATIVE', 'score': 0.9772131443023682}], [{'label': 'NEGATIVE', 'score': 0.9966067671775818}], [{'label': 'POSITIVE', 'score': 0.9792295098304749}]]
    assert result == answer


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

def test_qa_test_albert():
    """
    Tests
    HappyQuestionAnswering.test()
    """
    happy_tc = HappyTextClassification(model_type="ALBERT", model_name="textattack/albert-base-v2-SST-2")

    result = happy_tc.test("../data/tc/test.csv")
    answer = [[{'label': 'LABEL_1', 'score': 0.9990348815917969}], [{'label': 'LABEL_0', 'score': 0.9947203397750854}], [{'label': 'LABEL_0', 'score': 0.9958302974700928}], [{'label': 'LABEL_1', 'score': 0.9986426830291748}]]
    assert result == answer


def test_qa_train_effectiveness_albert():
    """
    Tests
    Ensures that HappyQuestionAnswering.train() results in
    lowering the loss as determined by HappyQuestionAnswering.eval()
    """

    happy_tc = HappyTextClassification(model_type="ALBERT", model_name="textattack/albert-base-v2-SST-2")
    before_loss = happy_tc.eval("../data/tc/train-eval.csv")["eval_loss"]
    happy_tc.train("../data/tc/train-eval.csv")
    after_loss = happy_tc.eval("../data/tc/train-eval.csv")["eval_loss"]
    assert after_loss < before_loss


def test_qa_test_bert():
    """
    Tests
    HappyQuestionAnswering.test()
    """
    happy_tc = HappyTextClassification(model_type="BERT", model_name="textattack/bert-base-uncased-SST-2")

    result = happy_tc.test("../data/tc/test.csv")
    answer = [[{'label': 'LABEL_1', 'score': 0.9995690584182739}], [{'label': 'LABEL_0', 'score': 0.9981549382209778}], [{'label': 'LABEL_0', 'score': 0.9965545535087585}], [{'label': 'LABEL_1', 'score': 0.9978235363960266}]]
    assert result == answer


def test_qa_train_effectiveness_bert():
    """
    Tests
    Ensures that HappyQuestionAnswering.train() results in
    lowering the loss as determined by HappyQuestionAnswering.eval()
    """

    happy_tc = HappyTextClassification(model_type="BERT", model_name="textattack/bert-base-uncased-SST-2")
    before_loss = happy_tc.eval("../data/tc/train-eval.csv")["eval_loss"]
    happy_tc.train("../data/tc/train-eval.csv")
    after_loss = happy_tc.eval("../data/tc/train-eval.csv")["eval_loss"]
    assert after_loss < before_loss
