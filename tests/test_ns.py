from happytransformer import HappyNextSentence


def test_sp_true():
    happy_ns = HappyNextSentence("BERT", "prajjwal1/bert-tiny")
    result = happy_ns.predict_next_sentence(
        "Hi nice to meet you. How old are you?",
        "I am 21 years old."
    )
    assert result > 0.5


def test_sp_false():
    happy_ns = HappyNextSentence("BERT", "prajjwal1/bert-tiny")
    result = happy_ns.predict_next_sentence(
        "How old are you?",
        "The Eiffel Tower is in Paris."
    )
    assert result < 0.5

def test_sp_save():
    happy = HappyNextSentence("BERT", "prajjwal1/bert-tiny")
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

