from happytransformer import HappyBERT

sentence_pairs = [
    ["How old are you?","The Eiffel Tower is in Paris"],
    ["How old are you?","I am 40 years old"]
]

happy = HappyBERT()
for a,b in sentence_pairs:
    print(a)
    print(b)
    print(happy.next_sentence_probability(a,b))