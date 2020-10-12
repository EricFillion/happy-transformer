from happytransformer import HappyBERT

# formats a float as a percentage
def percent(x):
    return f"{x*100:.2f}%"

# asserts that a function does throw an error
def assert_errors(func):
    try:
        func()
    except:
        return
    raise AssertionError()

happy = HappyBERT()
sentence_pairs = [
    ["How old are you?","The Eiffel Tower is in Paris",False],
    ["How old are you?","I am 40 years old",True]
]

def test_argument_errors():
    two_sentences = "This is the first sentence. This is the second sentence"
    one_sentence = "This is one sentence."
    assert_errors(lambda: happy.predict_next_sentence(two_sentences,one_sentence))
    assert_errors(lambda: happy.predict_next_sentence(one_sentence,two_sentences))
test_argument_errors()

def test_nsp():
    for a,b,follows in sentence_pairs:
        print('==============================')
        print(a)
        print(b)
        predict,probabilities = happy.predict_next_sentence(a,b)
        for p in probabilities:
            assert p>=0 and p<=1
        assert predict == follows
        print(predict,probabilities)
test_nsp()