'''
tests next sentence prediction capabilities
'''

from happytransformer import HappyBERT

def percent(x):
    '''formats a float as a percentage'''
    return f"{x*100:.2f}%"

def eq_ish(x, y, epsilon):
    '''soft similarity check between two numbers'''
    return abs(y-x) < epsilon

def errors(func):
    '''determines whether function errors'''
    try:
        func()
    except:
        return True
    return False

happy = HappyBERT()
SENTENCE_PAIRS = [
    ["How old are you?", "The Eiffel Tower is in Paris", False],
    ["How old are you?", "I am 40 years old", True]
]

def test_argument_errors():
    '''
    tests that the nsp module correctly rejects
    multi-sentence inputs
    '''
    two_sentences = "This is the first sentence. This is the second sentence"
    one_sentence = "This is one sentence."
    assert errors(lambda: happy.predict_next_sentence(two_sentences, one_sentence))
    assert errors(lambda: happy.predict_next_sentence(one_sentence, two_sentences))
test_argument_errors()

def test_nsp():
    '''
    tests that the nsp module returns expected results
    for the given sentence pairs
    '''
    for a, b, follows in SENTENCE_PAIRS:
        print('==============================')
        print(a)
        print(b)
        predict = happy.predict_next_sentence(a, b)
        probability = happy.predict_next_sentence(a, b, use_probability=True)
        assert 0 <= probability <= 1
        assert predict == follows
        print(predict, probability)
test_nsp()
