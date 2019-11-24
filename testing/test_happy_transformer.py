# disable pylint TODO warning
# pylint: disable=W0511, W0212

import pytest
""" Testing module for HappyBERT"""
from happy_transformer.happy_transformer import HappyTransformer


def test_text_verification():

    """
    Tests the method predict_mask in HappyTransformer()

    """
    model = HappyTransformer()

    test_string_mask = "Who was Jim Henson? Jim was a puppeteer."
    test_string_cls = "Who was Jim Henson? Jim [CLS] was a puppeteer."
    test_string_sep = "Who was Jim Henson? Jim [SEP] was a puppeteer."
    test_string_correct = "Who was Jim Henson? Jim [MASK] was a puppeteer."
    valid_mask = model._text_verification(test_string_mask)
    valid_cls = model._text_verification(test_string_cls)
    valid_sep = model._text_verification(test_string_sep)
    valid_correct = model._text_verification(test_string_correct)
    assert valid_mask is False
    assert valid_cls is False
    assert valid_sep is False
    assert valid_correct is True

def main():
    test_text_verification()

main()