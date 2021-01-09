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
    answer = [WordPredictionResult(token_str='pepper', score=0.2664579749107361), WordPredictionResult(token_str='vinegar', score=0.08760260790586472)]

    assert result == answer

def test_mwp_targets():
    happy_mwp = HappyWordPrediction()
    result = happy_mwp.predict_mask(
        "Please pass the salt and [MASK]",
        targets=["water", "spices"]
    )
    answer = [WordPredictionResult(token_str='water', score=0.014856964349746704), WordPredictionResult(token_str='spices', score=0.009040987119078636)]
    assert result == answer
