from happytransformer import HappyNextSentence

# Note, some of the model's weights are randomly initialized
# So we can not rely on getting the same score each time
# we run a unit test.


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


