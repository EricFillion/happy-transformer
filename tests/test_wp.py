from pytest import approx

from happytransformer import HappyWordPrediction
from happytransformer.happy_word_prediction import WordPredictionResult



def test_mwp_basic():
    MODELS = [
        ('DISTILBERT', 'distilbert-base-uncased', 'pepper'),
        ('BERT', 'bert-base-uncased', '.'),
        ('ALBERT', 'albert-base-v2', 'garlic'),
        ('ROBERTA', "roberta-base", "pepper")
    ]
    for model_type, model_name, top_result in MODELS:
        happy_wp = HappyWordPrediction(model_type, model_name)
        results = happy_wp.predict_mask(
            "Please pass the salt and [MASK]",
        )
        result = results[0]
        assert result.token == top_result


def test_mwp_top_k():
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


def test_mwp_targets():
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

def test_mwp_train_basic():
    happy_wp = HappyWordPrediction('', 'distilroberta-base')
    happy_wp.train("../data/wp/train-eval.txt")

def test_mwp_eval_basic():
    happy_wp = HappyWordPrediction('', 'distilroberta-base')
    result = happy_wp.eval("../data/wp/train-eval.txt")
    assert type(result.loss) == float

def test_mwp_train_effectiveness_multi():
    happy_wp = HappyWordPrediction('', 'distilroberta-base')

    before_result = happy_wp.eval("../data/wp/train-eval.txt")

    happy_wp.train("../data/wp/train-eval.txt")
    after_result = happy_wp.eval("../data/wp/train-eval.txt")

    assert after_result.loss < before_result.loss

