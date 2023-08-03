import csv
from happytransformer import HappyGeneration, HappyTextToText, HappyWordPrediction, GENTrainArgs, WPTrainArgs, TTTrainArgs
from datasets import load_from_disk
from  tests import happy_gen, happy_wp, happy_tt


def test_gen_len_max_len_trun():
    train_data = "../data/gen/train-eval.csv"
    save_path = "data/gen/len-test/"
    max_length = 5

    args_max_len_truncation = GENTrainArgs(max_length=max_length,
                        num_train_epochs=1,
                        save_path=save_path)

    happy_gen.train(train_data, args=args_max_len_truncation)

    tok_data = load_from_disk(save_path)
    print(tok_data)

    for case in tok_data["train"]["input_ids"]:
        print(case)
        assert len(case) <= max_length

    for case in tok_data["eval"]["input_ids"]:
        print(case)
        assert len(case) <= max_length


def test_gen_len_max_len_pad():
    train_data = "../data/gen/train-eval.csv"
    save_path = "data/gen/len-test/"
    max_length = 32

    args_max_len_truncation = GENTrainArgs(max_length=max_length,
                                           num_train_epochs=1,
                                           save_path=save_path)

    happy_gen.train(train_data, args=args_max_len_truncation)

    tok_data = load_from_disk(save_path)
    print(tok_data)

    for case in tok_data["train"]["input_ids"]:
        print(case)
        assert len(case) == max_length

    for case in tok_data["eval"]["input_ids"]:
        print(case)
        assert len(case) == max_length


def test_wp_len_max_len_trun():
    train_data = "../data/wp/train-eval.csv"
    save_path = "data/wp/len-test/"
    max_length = 5

    args_max_len_truncation = WPTrainArgs(max_length=max_length,
                                           num_train_epochs=1,
                                           save_path=save_path)

    happy_wp.train(train_data, args=args_max_len_truncation)

    tok_data = load_from_disk(save_path)
    print(tok_data)

    for case in tok_data["train"]["input_ids"]:
        print(case)
        assert len(case) <= max_length

    for case in tok_data["eval"]["input_ids"]:
        print(case)
        assert len(case) <= max_length


def test_wp_len_max_len_pad():
    train_data = "../data/wp/train-eval.csv"
    save_path = "data/wp/len-test/"
    max_length = 32

    args_max_len_truncation = WPTrainArgs(max_length=max_length, num_train_epochs=1, save_path=save_path)

    happy_wp.train(train_data, args=args_max_len_truncation)

    tok_data = load_from_disk(save_path)
    print(tok_data)

    for case in tok_data["train"]["input_ids"]:
        print(case)
        assert len(case) == max_length

    for case in tok_data["eval"]["input_ids"]:
        print(case)
        assert len(case) == max_length


def test_tt_len_max_len_trun():
    train_data = "../data/tt/train-eval-grammar.csv"
    save_path = "data/tt/len-test/"
    max_input_length = 4
    max_output_length = 2

    args_max_len_truncation = TTTrainArgs(
                        max_input_length=max_input_length,
                        max_output_length=max_output_length,
                        num_train_epochs=1,
                        save_path=save_path)

    happy_tt.train(train_data, args=args_max_len_truncation)

    tok_data = load_from_disk(save_path)
    print(tok_data)

    for case in tok_data["train"]["input_ids"]:
        print(case)
        assert len(case) == max_input_length

    for case in tok_data["eval"]["input_ids"]:
        print(case)
        assert len(case) == max_input_length


    for case in tok_data["train"]["labels"]:
        print(case)
        assert len(case) == max_output_length

    for case in tok_data["eval"]["labels"]:
        print(case)
        assert len(case) == max_output_length