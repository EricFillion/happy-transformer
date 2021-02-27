"""
Tests for the question answering training, evaluating and testing functionality
"""

from happytransformer.happy_question_answering import HappyQuestionAnswering
from pytest import approx

def test_qa_answer_question():
    MODELS = [
        ('ALBERT', 'twmkn9/albert-base-v2-squad2'),
        ('ROBERTA', 'deepset/roberta-base-squad2'),
        ('BERT', 'mrm8488/bert-tiny-5-finetuned-squadv2')
    ]
    for model_type, model_name in MODELS:
        happy_qa = HappyQuestionAnswering(model_name=model_name, model_type=model_type)
        answers = happy_qa.answer_question("Today's date is January 8th 2021", "What is the date?", top_k=3)

        assert sum(answer.score for answer in answers) == approx(1, 0.1)
        assert all('January 8th' in answer.answer for answer in answers)


def test_qa_eval():
    happy_qa = HappyQuestionAnswering(
        model_type='DISTILBERT',
        model_name='distilbert-base-cased-distilled-squad'
    )
    result = happy_qa.eval("../data/qa/train-eval.csv")
    assert result.loss == approx(0.11738169193267822, 0.001)


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
    # use a non-fine-tuned model so we DEFINITELY get an improvement
    happy_qa = HappyQuestionAnswering('BERT', 'mrm8488/bert-tiny-5-finetuned-squadv2')
    before_loss = happy_qa.eval("../data/qa/train-eval.csv").loss
    happy_qa.train("../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../data/qa/train-eval.csv").loss

    assert after_loss < before_loss
