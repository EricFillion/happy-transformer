"""
Tests for the question answering training, evaluating and testing functionality
"""

from happytransformer.happy_question_answering import HappyQuestionAnswering


def test_qa_answer_question():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.answer_question("Today's date is January 8th 2021", "What is the date?")
    answer = {'score': 0.9696964621543884, 'start': 16, 'end': 32, 'answer': 'January 8th 2021'}
    assert result == answer


def test_qa_answer_question_top_k():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.answer_question("Today's date is January 8th 2021", "What is the date?", topk=3)
    answer = [{'score': 0.9696964621543884, 'start': 16, 'end': 32, 'answer': 'January 8th 2021'}, {'score': 0.02050216868519783, 'start': 16, 'end': 27, 'answer': 'January 8th'}, {'score': 0.005092293489724398, 'start': 16, 'end': 23, 'answer': 'January'}]
    assert result == answer

def test_qa_train():
    happy_qa = HappyQuestionAnswering()
    happy_qa.train("../data/qa/train-eval.csv")


def test_qa_eval():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.eval("../data/qa/train-eval.csv")
    assert result["eval_loss"] == 0.11738169193267822


def test_qa_test():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.test("../data/qa/test.csv")
    answer = [{'score': 0.9939756989479065, 'start': 0, 'end': 12, 'answer': 'October 31st'}, {'score': 0.967872679233551, 'start': 12, 'end': 25, 'answer': 'November 23rd'}]
    assert result == answer


def test_qa_train_effectiveness():
    """
    Ensures that HappyQuestionAnswering.train() results in
    lowering the loss as determined by HappyQuestionAnswering.eval()
    """

    happy_qa = HappyQuestionAnswering()
    before_loss = happy_qa.eval("../data/qa/train-eval.csv")["eval_loss"]
    happy_qa.train("../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../data/qa/train-eval.csv")["eval_loss"]

    assert after_loss < before_loss

def test_qa_train_effectiveness_albert():
    """
    Ensures that HappyQuestionAnswering.train() results in
    lowering the loss as determined by HappyQuestionAnswering.eval()
    """

    happy_qa = HappyQuestionAnswering("ALBERT", "twmkn9/albert-base-v2-squad2")
    before_loss = happy_qa.eval("../data/qa/train-eval.csv")["eval_loss"]
    happy_qa.train("../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../data/qa/train-eval.csv")["eval_loss"]

    assert after_loss < before_loss

def test_qa_test_albert():
    happy_qa = HappyQuestionAnswering("ALBERT", "twmkn9/albert-base-v2-squad2")
    result = happy_qa.test("../data/qa/test.csv")
    print(result)
    answer = [{'score': 0.988578736782074, 'start': 0, 'end': 12, 'answer': 'October 31st'}, {'score': 0.9833534359931946, 'start': 12, 'end': 25, 'answer': 'November 23rd'}]
    assert result == answer