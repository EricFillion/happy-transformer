from happy_transformer.happy_bert import HappyBERT

from testing.test_bert import TestBERT
from testing.test_roberta import TestRoBERTa
from happy_transformer.happy_roberta import HappyRoBERTa
from happy_transformer.happy_bert import HappyBERT
def main():
    """testing"""
     # testRoBERTa = TestRoBERTa()
     # testRoBERTa.test_predict_mask()
    roBerta = HappyRoBERTa()
    text = roBerta.finish_sentence(text="Humans are for ")
    print(text)

    bert = HappyBERT()
    text = bert.finish_sentence(text="Humans are for ")
    print(text)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Aborting')
