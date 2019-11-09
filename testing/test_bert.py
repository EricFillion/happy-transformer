# disable pylint TODO warning
# pylint: disable=W0511
# todo import pytest

""" Testing module for HappyBERT"""
from happy_transformer.happy_bert import HappyBERT


class TestBERT:
    """
    A class that contains test methods for HappyBERT
    """

    def __init__(self):
        self.model = HappyBERT()

    def test_predict_mask(self):

        """
        Tests the method predict_mask in HappyBERT()

        """
        # TODO make the return token from test_predict_mask a string
        # TODO create a test for probs
        test_string = "Who was Jim Henson? Jim [MASK] was a puppeteer."
        token, score = self.model.predict_mask(test_string)
        assert token == "henson"
        print("Success: \"" + token + "\" was predicted for the input "+"\"" + test_string + "\"")
