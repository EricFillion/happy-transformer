# disable pylint TODO warning
# pylint: disable=W0511
# todo import pytest

""" Testing module for HappyBERT"""
from happy_transformer.happy_bert import HappyBERT


def test_predict_mask():
    """
    Tests the method predict_mask in HappyBERT()

    """
    # TODO make the return token from test_predict_mask a string
    # TODO create a test for probs
    model = HappyBERT()
    test_string = "Who was Jim Henson? Jim [MASK] was a puppeteer."
    token, score = model.predict_mask(test_string)
    assert token == "henson"
    assert score > 0.8
