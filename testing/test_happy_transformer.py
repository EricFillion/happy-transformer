# disable pylint TODO warning
# pylint: disable=W0511
# todo import pytest

""" Testing module for HappyBERT"""
from happy_transformer.happy_transformer import HappyTransformer


def test_predict_mask():
    """
    Tests the method predict_mask in HappyTransformer()

    """
    model = HappyTransformer()

    test_string_mask = "Who was Jim Henson? Jim was a puppeteer."
    test_string_cls = "Who was Jim Henson? Jim [CLS] was a puppeteer."
    test_string_sep = "Who was Jim Henson? Jim {SEP] was a puppeteer."
    test_string_eer = "Who was Jim Henson? Jim ##eer was a puppeteer."
    test_string_correct = "Who was Jim Henson? Jim [MASK] was a puppeteer."
    valid_mask = model.text_verification(test_string_mask)
    valid_cls = model.text_verification(test_string_cls)
    valid_sep = model.text_verification(test_string_sep)
    valid_eer = model.text_verification(test_string_eer)
    valid_correct = model.text_verification(test_string_correct)
    assert valid_mask == False
    assert valid_cls == False
    assert valid_sep == False
    assert valid_eer == False
    assert valid_correct == True