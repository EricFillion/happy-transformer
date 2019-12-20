"""
We will finish HappyGPT2 in January 2020

# disable pylint TODO warning
# pylint: disable=W0511
\""" Testing module for HappyGPT2\"""
import pytest

from happy_transformer.happy_gpt2 import HappyGPT2
from testing.standard_test_data import test_data


@pytest.mark.xfail
@pytest.mark.parametrize("tests",
                         [test_data[0]['data'], test_data[1]['data']],
                         ids=[test_data[0]['id'], test_data[1]['id']])
def test_predict_mask(tests):
    \"""
    Tests the method predict_mask in HappyGPT2()
    :param tests: A set of 10 tests to run
    \"""
    # TODO make the return token from test_predict_mask a string
    # TODO create a test for probs
    model = HappyGPT2()
    correct_words = 0
    correct_scores = 0
    incorrect_sentences = []
    for test in tests:
        str, options, expected_token, expected_score_threshold =\
            test
        token_dct = model.predict_mask(str, options)
        if token_dct[0]['word'] == expected_token:
            correct_words += 1
        else:
            incorrect_sentences.append(str)
        if token_dct[0]['score'] >= expected_score_threshold:
            correct_scores += 1

    assert correct_words >= 5
    assert correct_scores >= 5
    if correct_words < 5:
        print('Incorrect sentences:')
        for sentence in incorrect_sentences:
            print(sentence)
"""