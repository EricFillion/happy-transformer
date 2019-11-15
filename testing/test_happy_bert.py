# disable pylint TODO warning
# pylint: disable=W0511
""" Testing module for HappyBERT"""
import pytest

from happy_transformer.happy_bert import HappyBERT
from testing.standard_test_data import test_data


@pytest.mark.parametrize("test_string, options, expected_token, "
                         "expected_score_threshold",
                         [test_data[0]['data'], test_data[1]['data']],
                         ids=[test_data[0]['id'], test_data[1]['id']])
def test_predict_mask(test_string, options, expected_token,
                      expected_score_threshold):
    """
    Tests the method predict_mask in HappyBERT()

    """
    # TODO make the return token from test_predict_mask a string
    # TODO create a test for probs
    model = HappyBERT()
    token_dct = model.predict_mask(test_string, options)
    assert token_dct[0]['word'] == expected_token
    assert token_dct[0]['score'] > expected_score_threshold
