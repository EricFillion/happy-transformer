from happytransformer import HappyNextSentence


def test_sp_true():
    happy_ns = HappyNextSentence()
    result = happy_ns.predict_next_sentence(
        "Hi nice to meet you. How old are you?",
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

def test_sp_save():
    happy = HappyNextSentence()
    happy.save("model/")
    result_before = happy.predict_next_sentence(
        "How old are you?",
        "The Eiffel Tower is in Paris."
    )

    happy = HappyNextSentence(load_path="model/")
    result_after = happy.predict_next_sentence(
        "How old are you?",
        "The Eiffel Tower is in Paris."
    )

    assert result_before == result_after

