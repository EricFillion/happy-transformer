# disable pylint TODO warning
# pylint: disable=W0511

""" Testing module for HappyBERT"""
from happy_transformer.happy_roberta import HappyRoBERTa


class TestRoBERTa:
    """
    A class that contains test methods for HappyBERT
    """

    def __init__(self):

        self.model = HappyRoBERTa()

    def test_predict_mask(self):

        """
        Tests the method predict_mask in HappyRoBERTa()

        """

        test_string = "The first Star wars movie came out in <mask>"
        token, score = self.model.predict_mask(test_string)
        assert token == "1977"
        print("\ntest_predict_mask")
        print("Success: \"" + token + "\" was predicted for the input "+"\"" + test_string + "\"")

    def test_wsc(self):
        text = 'The city councilmen refused the demonstrators a permit because [they] feared violence.'
        output = self.model.wsc(text)

        assert output == "The city councilmen"
        print("\ntest_wsc")
        print("Success: \"" + output + "\" was predicted for the input "+"\"" + text + "\"")
