from happy_transformer.happy_xlnet import HappyXLNET
from happy_transformer.happy_bert import HappyBERT
from happy_transformer.happy_roberta import HappyRoBERTa


def main():
    """testing"""
    print("BERT")
    b = HappyBERT()
    print("Next sentence: ", b.is_next_sentence("I like cars.", "I also like cars."))
    print("1", b.predict_mask("My favourite food is [MASK]"))
    print("2", b.predict_mask("My favourite food is [MASK]", k=3))
    print("3", b.predict_mask("My favourite food is [MASK]", options=['rice', "ham", "apple"]))
    print("4", b.predict_mask("My favourite food is [MASK]", options=['rice', "ham", "apple"], k=1))
    print("5", b.predict_mask("My favourite food is [MASK]", options=['rice', "ham", "apple"], k=4))



    print("XLNET")
    xl = HappyXLNET()
    print("1", xl.predict_mask("My favourite food is [MASK]"))
    print("2", xl.predict_mask("My favourite food is [MASK]", k=3))
    print("3", xl.predict_mask("My favourite food is [MASK]", options=['rice', "ham", "apple"]))
    print("4", xl.predict_mask("My favourite food is [MASK]", options=['rice', "ham", "apple"], k=1))
    print("5", xl.predict_mask("My favourite food is [MASK]", options=['rice', "ham", "apple"], k=4))

    print("ROBERTA")
    hb = HappyRoBERTa()
    print("1", hb.predict_mask("My favourite food is [MASK]"))
    print("2", hb.predict_mask("My favourite food is [MASK]", k=3))
    print("3", hb.predict_mask("My favourite food is [MASK]", options=['rice', "ham", "apple"]))
    print("4", hb.predict_mask("My favourite food is [MASK]", options=['rice', "ham", "apple"], k=1))
    print("5", hb.predict_mask("My favourite food is [MASK]", options=['rice', "ham", "apple"], k=4))


    # print(hx.finish_sentence("Humans are for "))

    # a = hx.predict_mask("Hi, can I help you [MASK]")
    # print(a)
    #
    # hx.init_sequence_classifier()
    # hx.train_sequence_classifier("data/train.csv")
    # print(hx.eval_sequence_classifier("data/eval.csv"))
    # print(hx.test_sequence_classifier("data/test.csv"))
    #


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Aborting')
