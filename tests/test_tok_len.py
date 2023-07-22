import csv
from happytransformer import HappyGeneration, HappyWordPrediction,  GENTrainArgs, WPTrainArgs
from datasets import load_from_disk


def test_len_padding_max():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")

    train_data = "../data/gen/train-eval.csv"
    save_path = "data/gen/decode-test/"

    args= GENTrainArgs(num_train_epochs=1,
                       padding="max_length",  # default
                       truncation=True,  # default
                       max_length=None, # default
                       save_preprocessed_data=True,
                       save_preprocessed_data_path=save_path)

    happy_gen.train(train_data, args=args)

    tok_data = load_from_disk(save_path)


    for input_ids, labels  in zip(tok_data["train"]["input_ids"], tok_data["train"]["input_ids"]):
       assert len(input_ids) == len(labels)
       assert happy_gen.tokenizer.model_max_length == len(input_ids)


def test_len_padding_value():
    max_length = 5
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")

    train_data = "../data/gen/train-eval.csv"
    save_path = "data/gen/decode-test/"

    args= GENTrainArgs(num_train_epochs=1,
                       padding="max_length", # default
                       truncation=True,  # default
                       max_length=max_length,
                       save_preprocessed_data=True, # default
                       batch_size=2,
                       save_preprocessed_data_path=save_path)

    happy_gen.train(train_data, args=args)

    tok_data = load_from_disk(save_path)


    for input_ids, labels  in zip(tok_data["train"]["input_ids"], tok_data["train"]["input_ids"]):
       assert len(input_ids) == len(labels)
       assert max_length == len(input_ids)


def test_batch_size_gen():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")

    train_data = "../data/gen/train-eval.csv"
    save_path = "data/gen/decode-test/"

    # Batching should work by default due to all inputs being the
    # same length
    args= GENTrainArgs(num_train_epochs=1,
                       batch_size=2)

    happy_gen.train(train_data, args=args)




def test_batch_size_wp():
    happy_wp = HappyWordPrediction('BERT', 'bert-base-uncased')

    train_data = "../data/wp/train-eval.csv"

    # Batching should work by default due to all inputs being the
    # same length
    args= WPTrainArgs(num_train_epochs=1,
                       batch_size=2)

    happy_wp.train(train_data, args=args)

