from happytransformer import HappyWordPrediction
from happytransformer.happy_word_prediction import WordPredictionResult


def test_mwp_basic():
    happy_mwp = HappyWordPrediction()
    result = happy_mwp.predict_mask(
        "Please pass the salt and [MASK]",
    )
    answer = [WordPredictionResult(token_str="pepper", score=0.2664579749107361)]
    assert result == answer


def test_mwp_top_k():
    happy_mwp = HappyWordPrediction()
    result = happy_mwp.predict_mask(
        "Please pass the salt and [MASK]",
        top_k=2
    )
    answer = [WordPredictionResult(token_str='pepper', score=0.2664579749107361),
              WordPredictionResult(token_str='vinegar', score=0.08760260790586472)]

    assert result == answer


def test_mwp_targets():
    happy_mwp = HappyWordPrediction()
    result = happy_mwp.predict_mask(
        "Please pass the salt and [MASK]",
        targets=["water", "spices"]
    )
    answer = [WordPredictionResult(token_str='water', score=0.014856964349746704),
              WordPredictionResult(token_str='spices', score=0.009040987119078636)]
    assert result == answer


def test_mwp_basic_albert():
    happy_mwp = HappyWordPrediction("ALBERT", "albert-base-v2")
    result = happy_mwp.predict_mask(
        "Please pass the salt and [MASK]",
    )
    answer = [WordPredictionResult(token_str='garlic', score=0.036625903099775314)]
    assert result == answer


def test_mwp_basic_bert():
    happy_mwp = HappyWordPrediction("BERT", "bert-base-uncased")
    result = happy_mwp.predict_mask(
        "Please pass the salt and [MASK]",
    )
    answer = [WordPredictionResult(token_str='.', score=0.8466101884841919)]
    assert result == answer


def test_mwp_basic_roberta():
    happy_mwp = HappyWordPrediction("ROBERTA", "roberta-base")
    result = happy_mwp.predict_mask(
        "Please pass the salt and [MASK]",
    )
    answer = [WordPredictionResult(token_str='pepper', score=0.7325230240821838)]
    assert result == answer
