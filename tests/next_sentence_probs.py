from happytransformer import HappyBERT

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

def nsp_with_multiple_sentences():
    happy.next_sentence_probability(
        "This is the first sentence. This is the second sentence",
        "An error should be raised here"
    )
assert_errors(nsp_with_multiple_sentences)

for a,b,follows in sentence_pairs:
    print('==============================')
    print(a)
    print(b)
    p = happy.next_sentence_probability(a,b)
    assert p>=0 and p<=1
    predict_follows = p > 0.5
    assert predict_follows == follows
    print(percent(p))