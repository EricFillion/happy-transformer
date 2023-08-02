from pytest import approx

from happytransformer import (
    HappyWordPrediction,
    WPTrainArgs,
    WPEvalArgs
)
from happytransformer.happy_word_prediction import WordPredictionResult
from tests.run_save_load import run_save_load
import pytest

def test_wp_basic():
    MODELS = [
        ('DISTILBERT', 'distilbert-base-uncased', 'pepper'),
        ('BERT', 'bert-base-uncased', '.'),
        ('ALBERT', 'albert-base-v2', 'garlic'),
        ('ROBERTA', "roberta-base", ' pepper')  # todo look into why roberta predicts a space
    ]
    for model_type, model_name, top_result in MODELS:
        happy_wp = HappyWordPrediction(model_type, model_name)
        results = happy_wp.predict_mask(
            "Please pass the salt and [MASK]",
        )
        result = results[0]
        assert result.token == top_result


def test_wp_high_k():

    happy_wp = HappyWordPrediction("DISTILBERT", "distilbert-base-uncased")
    results = happy_wp.predict_mask(
        "Please pass the salt and [MASK]", top_k=3000
    )
    assert results[0].token == "pepper"

def test_wp_top_k():
    happy_wp = HappyWordPrediction('DISTILBERT', 'distilbert-base-uncased')
    result = happy_wp.predict_mask(
        "Please pass the salt and [MASK]",
        top_k=2
    )
    answer = [
        WordPredictionResult(token='pepper', score=approx(0.2664579749107361, 0.01)),
        WordPredictionResult(token='vinegar', score=approx(0.08760260790586472, 0.01))
    ]

    assert result == answer


def test_wp_targets():
    happy_wp = HappyWordPrediction('DISTILBERT', 'distilbert-base-uncased')
    result = happy_wp.predict_mask(
        "Please pass the salt and [MASK]",
        targets=["water", "spices"], top_k=2
    )
    assert result[0].token == "water" and result[1].token == "spices"
    assert type(result[0].score) == float
    assert 0.0138 <= result[0].score <= 0.0158


def test_wp_train_default():
    happy_wp = HappyWordPrediction('BERT', 'prajjwal1/bert-tiny')
    happy_wp.train("../data/wp/train-eval.txt", args = WPTrainArgs(max_length=512))

def test_wp_train_line_by_line():
    happy_wp = HappyWordPrediction('BERT', 'prajjwal1/bert-tiny')
    happy_wp.train("../data/wp/train-eval.txt", args=WPTrainArgs(line_by_line=True, max_length=512))



def test_wp_eval_basic():
    happy_wp = HappyWordPrediction('BERT', 'prajjwal1/bert-tiny')
    result = happy_wp.eval("../data/wp/train-eval.txt", args=WPEvalArgs(max_length=512))
    assert type(result.loss) == float

def test_wp_train_effectiveness_multi():
    happy_wp = HappyWordPrediction('BERT', 'prajjwal1/bert-tiny')

    before_result = happy_wp.eval("../data/wp/train-eval.txt", args=WPEvalArgs(max_length=512))

    happy_wp.train("../data/wp/train-eval.txt", args=WPTrainArgs(max_length=512))
    after_result = happy_wp.eval("../data/wp/train-eval.txt", args=WPEvalArgs(max_length=512))

    assert after_result.loss < before_result.loss

def test_wp_eval_some_settings():
    """
    Test to see what happens when only a subset of the potential settings are used
    :return:
    """
    args = {'line_by_line': True, "max_length": 512}

    happy_wp = HappyWordPrediction('BERT', 'prajjwal1/bert-tiny')

    with pytest.raises(ValueError):
        result = happy_wp.eval("../data/wp/train-eval.txt", args)



def test_wp_save_load_train():
    happy_wp = HappyWordPrediction('BERT', 'prajjwal1/bert-tiny')
    output_path = "data/wp-train/"
    data_path = "../data/wp/train-eval.txt"
    args = WPTrainArgs(line_by_line=True, max_length=512)
    run_save_load(happy_wp, output_path, args, data_path, "train")

def test_wp_save_load_eval():
    happy_wp = HappyWordPrediction('BERT', 'prajjwal1/bert-tiny')
    output_path = "data/wp-eval.json"
    data_path = "../data/wp/train-eval.txt"
    args = WPEvalArgs(line_by_line=True, max_length=512)
    run_save_load(happy_wp, output_path, args, data_path, "eval")

def test_wp_save():
    happy = HappyWordPrediction("BERT", "prajjwal1/bert-tiny")
    happy.save("model/")
    result_before = happy.predict_mask("I think therefore I [MASK]")

    happy = HappyWordPrediction(model_name="model/")
    result_after = happy.predict_mask("I think therefore I [MASK]")

    assert result_before[0].token ==result_after[0].token


def test_wp_train_eval_with_dic():

    happy_wp = HappyWordPrediction('BERT', 'prajjwal1/bert-tiny')
    train_args = {'learning_rate': 0.01, 'line_by_line': True, "num_train_epochs": 1, "max_length": 512}

    # dictionaries are no longer supported so we expect a ValueError
    with pytest.raises(ValueError):
        happy_wp.train("../data/wp/train-eval.txt" , args=train_args)

    eval_args = {'line_by_line': True}
    with pytest.raises(ValueError):
        after_result = happy_wp.eval("../data/wp/train-eval.txt", args=eval_args)


def test_wp_train_eval_with_dataclass():

    happy_wp = HappyWordPrediction('BERT', 'prajjwal1/bert-tiny')
    train_args = WPTrainArgs(learning_rate=0.01, line_by_line=True, num_train_epochs=1, max_length=512)

    happy_wp.train("../data/wp/train-eval.txt" , args=train_args)

    eval_args = WPEvalArgs(line_by_line=True, max_length=512)

    after_result = happy_wp.eval("../data/wp/train-eval.txt", args=eval_args)

def test_wp_csv():
    data_path = "../data/wp/train-eval.csv"

    mlm_probability = 0.5  # set high due to this issue https://github.com/huggingface/transformers/issues/16711

    happy_wp = HappyWordPrediction('BERT', 'bert-base-uncased')
    before_result = happy_wp.eval(data_path, args=WPEvalArgs(mlm_probability=mlm_probability))
    print("before_result", before_result)
    happy_wp.train(data_path, args=WPTrainArgs(mlm_probability=mlm_probability))
    after_result = happy_wp.eval(data_path, args=WPEvalArgs(mlm_probability=mlm_probability))
    print("after_result", after_result)

    assert after_result.loss < before_result.loss
