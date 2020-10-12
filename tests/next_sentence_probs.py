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
    assert_errors(lambda: happy.next_sentence_probability(one_sentence,two_sentences))
    assert_errors(lambda: happy.next_sentence_probability(two_sentences,one_sentence))
test_argument_errors()

def test_nsp():
    for a,b,follows in sentence_pairs:
        print('==============================')
        print(a)
        print(b)
        p = happy.next_sentence_probability(a,b)
        assert p>=0 and p<=1
        predict_follows = p > 0.5
        assert predict_follows == follows

        assert happy.predict_next_sentence(a,b) == follows
        print(percent(p))
test_nsp()