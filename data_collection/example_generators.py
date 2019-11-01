"""

Contains modules to generate examples for both testing and training

"""

import re
from data_collection.data import data_collection


def fit_bert_test_generator():
    """
    Uses the wsc 278 to generate testing examples that can be used with fitBERT.
    fitBERT requires that the masked word be labeled "***mask***"
    The output is saved to a csv file called "fit_bert_test_generator"
    """

    df = data_collection.get_data()

    masked_sentences = list()
    for index, row in df.iterrows():
        masked_sentence = row['txt1'] + " ***mask***" + row['txt2']
        masked_sentence = masked_sentence.replace("\n", " ")
        masked_sentence = re.sub(' +', ' ', masked_sentence)
        masked_sentences.append(masked_sentence)

    df["masked_sentences"] = masked_sentences
    new_df = df[['masked_sentences', 'OptionA', 'OptionB', "answer"]].copy()

    new_df.to_csv('wsc_test_fitbert.csv', index=None, header=True)

    print("dataset successfully saved to \"wsc_test_fitbert.csv\" ")

