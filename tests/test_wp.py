from pytest import approx

from happytransformer import (
    HappyWordPrediction,
    ARGS_WP_TRAIN,
    ARGS_WP_EVAl,
    WPTrainArgs,
    WPEvalArgs
)
from happytransformer.happy_word_prediction import WordPredictionResult
from tests.shared_tests import run_save_load


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
        targets=["water", "spices"]
    )
    answer = [
        WordPredictionResult(token='water', score=approx(0.014856964349746704, 0.01)),
        WordPredictionResult(token='spices', score=approx(0.009040987119078636, 0.01))
    ]
    assert result == answer

def test_wp_train_default():
    happy_wp = HappyWordPrediction('', 'distilroberta-base')
    happy_wp.train("../data/wp/train-eval.txt")

def test_wp_train_line_by_line():
    happy_wp = HappyWordPrediction('', 'distilroberta-base')
    happy_wp.train("../data/wp/train-eval.txt", args=WPTrainArgs(line_by_line=True))



def test_wp_eval_basic():
    happy_wp = HappyWordPrediction('', 'distilroberta-base')
    result = happy_wp.eval("../data/wp/train-eval.txt")
    assert type(result.loss) == float

def test_wp_train_effectiveness_multi():
    happy_wp = HappyWordPrediction('', 'distilroberta-base')

    before_result = happy_wp.eval("../data/wp/train-eval.txt")

    happy_wp.train("../data/wp/train-eval.txt")
    after_result = happy_wp.eval("../data/wp/train-eval.txt")

    assert after_result.loss < before_result.loss

def test_wp_eval_some_settings():
    """
    Test to see what happens when only a subset of the potential settings are used
    :return:
    """
    args = {'line_by_line': True,
            }
    happy_wp = HappyWordPrediction('', 'distilroberta-base')
    result = happy_wp.eval("../data/wp/train-eval.txt", args)
    assert type(result.loss) == float


def test_wp_save_load_train():
    happy_wp = HappyWordPrediction('', 'distilroberta-base')
    output_path = "data/wp-train.json"
    data_path = "../data/wp/train-eval.txt"
    args = ARGS_WP_TRAIN
    args["line_by_line"] = True
    run_save_load(happy_wp, output_path, args, data_path, "train")

def test_wp_save_load_eval():
    happy_wp = HappyWordPrediction('', 'distilroberta-base')
    output_path = "data/wp-eval.json"
    data_path = "../data/wp/train-eval.txt"
    args = ARGS_WP_EVAl
    args["line_by_line"] = True
    run_save_load(happy_wp, output_path, args, data_path, "eval")

def test_wp_save():
    happy = HappyWordPrediction("BERT", "prajjwal1/bert-tiny")
    happy.save("model/")
    result_before = happy.predict_mask("I think therefore I [MASK]")

    happy = HappyWordPrediction(load_path="model/")
    result_after = happy.predict_mask("I think therefore I [MASK]")

    assert result_before[0].token ==result_after[0].token


def test_wp_train_eval_with_dic():

    happy_wp = HappyWordPrediction('', 'distilroberta-base')
    train_args = {'learning_rate': 0.01, 'line_by_line': True, "num_train_epochs": 1}


    happy_wp.train("../data/wp/train-eval.txt" , args=train_args)
    eval_args = {'line_by_line': True}

    after_result = happy_wp.eval("../data/wp/train-eval.txt", args=eval_args)


def test_wp_train_eval_with_dataclass():

    happy_wp = HappyWordPrediction('', 'distilroberta-base')
    train_args = WPTrainArgs(learning_rate=0.01, line_by_line=True, num_train_epochs=1)

    happy_wp.train("../data/wp/train-eval.txt" , args=train_args)

    eval_args = WPEvalArgs(line_by_line=True)

    after_result = happy_wp.eval("../data/wp/train-eval.txt", args=eval_args)

