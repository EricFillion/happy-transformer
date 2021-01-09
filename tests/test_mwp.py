from happytransformer import HappyWordPrediction


def test_mwp_basic():
    happy_mwp = HappyWordPrediction()
    result = happy_mwp.predict_masks(
        "Please pass the salt and [MASK]",
    )
    answer = {'score': 0.2664579749107361, 'token_str': 'pepper'}
    assert result == answer


def test_mwp_top_k():
    happy_mwp = HappyWordPrediction()
    result = happy_mwp.predict_masks(
        "Please pass the salt and [MASK]",
        top_k=2
    )
    answer = [{'score': 0.2664579749107361, 'token_str': 'pepper'}, {'score': 0.08760260790586472, 'token_str': 'vinegar'}]

    assert result == answer

def test_mwp_targets():
    happy_mwp = HappyWordPrediction()
    result = happy_mwp.predict_masks(
        "Please pass the salt and [MASK]",
        targets=["water", "spices"]
    )
    answer = {'score': 0.014856964349746704, 'token_str': 'water'}
    assert result == answer


def test_mwp_train():
    happy_mwp = HappyWordPrediction()
    happy_mwp.train()
