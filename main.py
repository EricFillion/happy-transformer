from happy_transformer.happy_bert import HappyBERT

from testing.test_bert import TestBERT
from testing.test_roberta import TestRoBERTa
def main():
    """testing"""
    testRoBERTa = TestRoBERTa()
    testRoBERTa.test_predict_mask()

    testBERT = TestBERT()
    testBERT.test_predict_mask()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Aborting')
