from happy_transformer.happy_xlnet import HappyXLNET
from happy_transformer.happy_bert import HappyBERT


def main():
    """testing"""
    hb = HappyBERT()
    # print(hb.is_next_sentence("I like cars.", "I also like cars."))
    print(hb.answer_question("What is the meaning of life?", "philosophy"))

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
