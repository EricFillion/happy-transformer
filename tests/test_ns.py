from happytransformer import HappyNextSentence
from tests import happy_ns

def test_sp_true():
    result = happy_ns.predict_next_sentence(
        "Hi nice to meet you. How old are you?",
        "I am 21 years old."
    )
    assert result > 0.5


def test_sp_false():
    result = happy_ns.predict_next_sentence(
        "How old are you?",
        "The Eiffel Tower is in Paris."
    )
    assert result < 0.5

def test_sp_save():
    happy_ns.save("model/")
    result_before = happy_ns.predict_next_sentence(
        "How old are you?",
        "The Eiffel Tower is in Paris."
    )

    happy = HappyNextSentence("BERT", model_name="model/")
    result_after = happy.predict_next_sentence(
        "How old are you?",
        "The Eiffel Tower is in Paris."
    )

    assert result_before == result_after

