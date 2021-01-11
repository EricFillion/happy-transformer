"""
Tests for the question answering training, evaluating and testing functionality
"""

from happytransformer.happy_question_answering import HappyQuestionAnswering, QuestionAnsweringResult


def test_qa_answer_question():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.answer_question("Today's date is January 8th 2021", "What is the date?")
    answer = [QuestionAnsweringResult(answer='January 8th 2021', score=0.9696964621543884, start=16, end=32)]
    assert result == answer


def test_qa_answer_question_top_k():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.answer_question("Today's date is January 8th 2021", "What is the date?", topk=3)
    answer = [QuestionAnsweringResult(answer='January 8th 2021', score=0.9696964621543884, start=16, end=32),
              QuestionAnsweringResult(answer='January 8th', score=0.02050216868519783, start=16, end=27),
              QuestionAnsweringResult(answer='January', score=0.005092293489724398, start=16, end=23)]
    assert result == answer


def test_qa_train():
    happy_qa = HappyQuestionAnswering()
    happy_qa.train("../data/qa/train-eval.csv")


def test_qa_eval():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.eval("../data/qa/train-eval.csv")
    assert result.eval_loss == 0.11738169193267822


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
    before_loss = happy_qa.eval("../data/qa/train-eval.csv").eval_loss
    happy_qa.train("../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../data/qa/train-eval.csv").eval_loss

    assert after_loss < before_loss


def test_qa_train_effectiveness_albert():
    """
    Ensures that HappyQuestionAnswering.train() results in
    lowering the loss as determined by HappyQuestionAnswering.eval()
    """

    happy_qa = HappyQuestionAnswering("ALBERT", "twmkn9/albert-base-v2-squad2")
    before_loss = happy_qa.eval("../data/qa/train-eval.csv").eval_loss
    happy_qa.train("../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../data/qa/train-eval.csv").eval_loss

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
    before_loss = happy_qa.eval("../data/qa/train-eval.csv").eval_loss
    happy_qa.train("../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../data/qa/train-eval.csv").eval_loss

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

    happy_qa = HappyQuestionAnswering("ROBERTA", "roberta-base")
    before_loss = happy_qa.eval("../data/qa/train-eval.csv").eval_loss
    happy_qa.train("../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../data/qa/train-eval.csv").eval_loss
    assert after_loss < before_loss


def test_qa_test_roberta():
    happy_qa = HappyQuestionAnswering("ROBERTA", "roberta-base")
    result = happy_qa.test("../data/qa/test.csv")
    print(result)
    answer = [QuestionAnsweringResult(answer='is the', score=0.03888237848877907, start=13, end=19),
              QuestionAnsweringResult(answer='date is', score=0.02540113404393196, start=4, end=11)]


if __name__ == '__main__':
    test_qa_test_roberta()
    test_qa_train_effectiveness_roberta()
