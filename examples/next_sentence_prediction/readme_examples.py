from happytransformer import HappyNextSentence

def example_4_0():
    happy_ns = HappyNextSentence()  # default is "bert-base-uncased"
    happy_ns_large = HappyNextSentence("BERT", "bert-large-uncased")


def example_4_1():
    happy_ns = HappyNextSentence()
    result = happy_ns.predict_next_sentence(
        "How old are you?",
        "I am 21 years old."
    )
    print(type(result))  # <class 'float'>
    print(result)  # 0.9999918937683105

def example_4_2():
    happy_ns = HappyNextSentence()
    result = happy_ns.predict_next_sentence(
        "How old are you?",
        "Queen's University is in Kingston Ontario Canada"
    )
    print(type(result))  # <class 'float'>
    print(result)  # 0.00018497584096621722


def main():
    example_4_2()


if __name__ == "__main__":
    main()
