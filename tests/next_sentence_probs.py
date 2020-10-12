from happytransformer import HappyBERT

def percent(x):
    return f"{x*100:.2f}%"

sentence_pairs = [
    ["How old are you?","The Eiffel Tower is in Paris",False],
    ["How old are you?","I am 40 years old",True]
]

happy = HappyBERT()
for a,b,follows in sentence_pairs:
    print('==============================')
    print(a)
    print(b)
    p = happy.next_sentence_probability(a,b)
    assert p>=0 and p<=1
    predict_follows = p > 0.5
    assert predict_follows == follows
    print(percent(p))