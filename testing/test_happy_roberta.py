# disable pylint TODO warning
# pylint: disable=W0511

""" Testing module for HappyBERT"""
import pytest

from happy_transformer.happy_roberta import HappyRoBERTa


def test_predict_mask():
    """
    Tests the method predict_mask in HappyRoBERTa()

    """
    model = HappyRoBERTa()
    test_string = "The first Star wars movie came out in <mask>"
    token, score = model.predict_mask(test_string)
    assert token == "1977"
    assert score > 0.8


@pytest.mark.xfail
def test_wsc():
    """
    Tests the method wsc in HappyRoBERTa()

    """
    model = HappyRoBERTa()
    text = 'The city councilmen refused the demonstrators a permit because '\
           '[they] feared violence.'
    output = model.wsc(text)

    assert output == "The city councilmen"
