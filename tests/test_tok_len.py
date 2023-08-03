import csv
from happytransformer import HappyWordPrediction,  GENTrainArgs, WPTrainArgs
from datasets import load_from_disk
from tests import happy_gen, happy_wp


def test_len_padding_max():

    train_data = "../data/gen/train-eval.csv"
    save_path = "data/gen/decode-test/"

    args= GENTrainArgs(num_train_epochs=1,
                       max_length=None, # default
                       save_path=save_path)

    happy_gen.train(train_data, args=args)

    tok_data = load_from_disk(save_path)


    for input_ids, labels  in zip(tok_data["train"]["input_ids"], tok_data["train"]["input_ids"]):
       assert len(input_ids) == len(labels)
       assert happy_gen.tokenizer.model_max_length == len(input_ids)


def test_len_padding_value():
    max_length = 5

    train_data = "../data/gen/train-eval.csv"
    save_path = "data/gen/decode-test/"

    args= GENTrainArgs(num_train_epochs=1,
                       max_length=max_length,
                       batch_size=2,
                       save_path=save_path)

    happy_gen.train(train_data, args=args)

    tok_data = load_from_disk(save_path)


    for input_ids, labels  in zip(tok_data["train"]["input_ids"], tok_data["train"]["input_ids"]):
       assert len(input_ids) == len(labels)
       assert max_length == len(input_ids)


def test_batch_size_gen():
    train_data = "../data/gen/train-eval.csv"
    save_path = "data/gen/decode-test/"

    # Batching should work by default due to all inputs being the
    # same length
    args= GENTrainArgs(num_train_epochs=1,
                       batch_size=2)

    happy_gen.train(train_data, args=args)




def test_batch_size_wp():
    train_data = "../data/wp/train-eval.csv"

    # Batching should work by default due to all inputs being the
    # same length
    args= WPTrainArgs(num_train_epochs=1,
                       batch_size=2)

    happy_wp.train(train_data, args=args)

