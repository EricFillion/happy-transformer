"""
Tests for methods under HappyTransformer.
"""
from happytransformer import HappyWordPrediction

def test_save():
    happy = HappyWordPrediction("BERT", "prajjwal1/bert-tiny")
    happy.save("model/")
    result_before= happy.predict_mask("I think therefore I [MASK]")

    happy = HappyWordPrediction(load_path="model/")
    result_after= happy.predict_mask("I think therefore I [MASK]")

    assert type(result_before[0].token ==result_after[0].token)




