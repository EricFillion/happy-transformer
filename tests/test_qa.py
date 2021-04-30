"""
Tests for the question answering training, evaluating and testing functionality
"""

from happytransformer.happy_question_answering import HappyQuestionAnswering
from happytransformer.qa.trainer import QATrainArgs, QAEvalArgs, QATestArgs
from happytransformer.qa.default_args import ARGS_QA_TRAIN, ARGS_QA_EVAl
from tests.shared_tests import run_save_load_train

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


def test_qa_train():
    happy_qa = HappyQuestionAnswering(
        model_type='DISTILBERT',
        model_name='distilbert-base-cased-distilled-squad'
    )
    result = happy_qa.train("../data/qa/train-eval.csv")


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


def test_qa_save():
    happy = HappyQuestionAnswering()
    happy.save("model/")
    result_before = happy.answer_question("Natural language processing is a subfield of artificial surrounding creating models that understand language","What is natural language processing?")

    happy = HappyQuestionAnswering(load_path="model/")
    result_after = happy.answer_question("Natural language processing is a subfield of artificial surrounding creating models that understand language","What is natural language processing?")

    assert result_before[0].answer == result_after[0].answer


def test_qa_with_dic():

    happy_qa = HappyQuestionAnswering()
    train_args = {'learning_rate': 0.01,  "num_train_epochs": 1}


    happy_qa.train("../data/qa/train-eval.csv" , args=train_args)

    eval_args = {}

    result_eval = happy_qa.eval("../data/qa/train-eval.csv", args=eval_args)
    assert result_eval.loss == approx(2.544920206069, 0.001)

    test_args = {}

    result_test = happy_qa.test("../data/qa/test.csv", args=test_args)
    assert result_test[0].answer == "October"

def test_tc_with_dataclass():

    happy_qa = HappyQuestionAnswering()
    train_args = QATrainArgs(learning_rate=0.01, num_train_epochs=1)

    happy_qa.train("../data/qa/train-eval.csv", args=train_args)

    eval_args = QAEvalArgs()

    result_eval = happy_qa.eval("../data/qa/train-eval.csv", args=eval_args)
    assert result_eval.loss == approx(2.544920206069, 0.001)


    test_args = QATestArgs()

    result_test = happy_qa.test("../data/qa/test.csv", args=test_args)
    assert result_test[0].answer == "October"


def test_tc_save_load_train():
    happy_wp = HappyQuestionAnswering()
    output_path = "data/qa-train.json"
    data_path = "../data/qa/train-eval.csv"
    run_save_load_train(happy_wp, output_path, ARGS_QA_TRAIN, data_path, "train")


def test_tc_save_load_eval():
    happy_wp = HappyQuestionAnswering()
    output_path = "data/qa-train.json"
    data_path = "../data/qa/train-eval.csv"
    run_save_load_train(happy_wp, output_path, ARGS_QA_EVAl, data_path, "train")
