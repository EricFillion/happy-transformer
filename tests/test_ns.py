from happytransformer import HappyNextSentence
import pytest
# Note, some of the model's weights are randomly initialized
# So we can not rely on getting the same score each time
# we run a unit test.


def test_sp_true():
    happy_ns = HappyNextSentence()
    result = happy_ns.predict_next_sentence(
        "How old are you?",
        "I am 21 years old."
    )
    assert result > 0.5


def test_sp_false():
    happy_ns = HappyNextSentence()
    result = happy_ns.predict_next_sentence(
        "How old are you?",
        "The Eiffel Tower is in Paris."
    )
    assert result < 0.5


def test_sp_sa_too_long():
    happy_ns = HappyNextSentence()
    with pytest.raises(Exception):
        result = happy_ns.predict_next_sentence(
            "How old are you? I'm 21 years old.",
            "I am 93 years old."
        )


def test_sp_sb_too_long():
    happy_ns = HappyNextSentence()
    with pytest.raises(Exception):
        result = happy_ns.predict_next_sentence(
            "How old are you?",
            "I am 93 years old. I'm 21 years old."
        )
