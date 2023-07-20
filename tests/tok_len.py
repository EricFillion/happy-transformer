import csv
from happytransformer import HappyGeneration, HappyTextToText,  GENTrainArgs, TTTrainArgs
from datasets import load_from_disk


def test_len_padding_max():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")

    train_data = "../data/gen/train-eval.csv"
    save_path = "data/gen/decode-test/"

    args= GENTrainArgs(num_train_epochs=1,
                       padding="max_length",
                       truncation=True,
                       max_length=None,
                       save_preprocessed_data=True,
                       save_preprocessed_data_path=save_path)

    happy_gen.train(train_data, args=args)

    tok_data = load_from_disk(save_path)


    for input_ids, labels  in zip(tok_data["train"]["input_ids"], tok_data["train"]["input_ids"]):
       assert len(input_ids) == len(labels)
       assert happy_gen.tokenizer.model_max_length == len(input_ids)

