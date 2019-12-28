
from happy_transformer.happy_xlnet import HappyXLNET
import pandas as pd

def main():
    """testing"""

    xl = HappyXLNET(model="xlnet-base-cased")


    train_df = pd.read_csv("data/train.csv", header=None)
    train_df[0] = (train_df[0] == 2).astype(int)
    train_df = pd.DataFrame({
        'id': range(len(train_df)),
        'label': train_df[0],
        'alpha': ['a'] * train_df.shape[0],
        'text': train_df[1].replace(r'\n', ' ', regex=True)
    })

    xl.init_sequence_classifier(classifier_name="xlnet-seq-class")
    xl.train_sequence_classifier(train_df=train_df, overwrite_output_dir=True)

    test_df = pd.read_csv("data/test.csv", header=None)
    test_df[0] = (test_df[0] == 2).astype(int)
    test_df = pd.DataFrame({
        'id': range(len(test_df)),
        'label': test_df[0],
        'alpha': ['a'] * test_df.shape[0],
        'text': test_df[1].replace(r'\n', ' ', regex=True)
    })
    results = xl.eval_sequence_classifier(test_df)

    print(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Aborting')
