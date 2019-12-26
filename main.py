from happy_transformer.happy_roberta import HappyRoBERTa
from happy_transformer.happy_bert import HappyBERT
from happy_transformer.happy_xlnet import HappyXLNET
import pandas as pd

def main():
    """testing"""

    xl = HappyXLNET(model="xlnet-base-cased")
    print("here")

    print(xl.predict_mask("Humans are for [MASK]"))
    print("here")

    xl = HappyBERT()
    train_df = pd.read_csv("data/train.csv", header=None)
    train_df[0] = (train_df[0] == 2).astype(int)
    train_df = pd.DataFrame({
        'id': range(len(train_df)),
        'label': train_df[0],
        'alpha': ['a'] * train_df.shape[0],
        'text': train_df[1].replace(r'\n', ' ', regex=True)
    })

    xl._init_sequence_classifier("bert_seq")
    xl._train_sequence_classifier(train_df)

    test_df = pd.read_csv("data/test.csv", header=None)
    test_df[0] = (test_df[0] == 2).astype(int)
    test_df = pd.DataFrame({
        'id': range(len(train_df)),
        'label': test_df[0],
        'alpha': ['a'] * train_df.shape[0],
        'text': test_df[1].replace(r'\n', ' ', regex=True)
    })
    xl._eval_sequence_classifier(test_df)




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Aborting')
