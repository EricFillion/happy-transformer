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
        happy_mwp = HappyWordPrediction(model_type, model_name)
        results = happy_mwp.predict_mask(
            "Please pass the salt and [MASK]",
        )
        result = results[0]
        assert result.token == top_result


def test_mwp_top_k():
    happy_mwp = HappyWordPrediction('DISTILBERT', 'distilbert-base-uncased')
    result = happy_mwp.predict_mask(
        "Please pass the salt and [MASK]",
        top_k=2
    )
    answer = [
        WordPredictionResult(token='pepper', score=approx(0.2664579749107361, 0.01)),
        WordPredictionResult(token='vinegar', score=approx(0.08760260790586472, 0.01))
    ]

    assert result == answer


def test_mwp_targets():
    happy_mwp = HappyWordPrediction('DISTILBERT', 'distilbert-base-uncased')
    result = happy_mwp.predict_mask(
        "Please pass the salt and [MASK]",
        targets=["water", "spices"]
    )
    answer = [
        WordPredictionResult(token='water', score=approx(0.014856964349746704, 0.01)),
        WordPredictionResult(token='spices', score=approx(0.009040987119078636, 0.01))
    ]
    assert result == answer
