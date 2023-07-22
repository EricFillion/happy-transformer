import csv
from happytransformer import HappyGeneration, HappyTextToText, HappyWordPrediction,  GENTrainArgs, TTTrainArgs, WPTrainArgs
from datasets import load_from_disk


def test_gen_csv_decode():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")

    train_data = "../data/gen/train-eval.csv"
    save_path = "data/gen/decode-test/"

    args_max_len_truncation = GENTrainArgs(num_train_epochs=1,
                                           save_preprocessed_data=True,
                                           save_preprocessed_data_path=save_path)

    happy_gen.train(train_data, args=args_max_len_truncation)

    tok_data = load_from_disk(save_path)

    lines = []
    with open(train_data, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lines.append(row["text"])

    for case in tok_data["train"]["input_ids"]:
        detok_case = happy_gen.tokenizer.decode(case, skip_special_tokens=True)
        assert detok_case in lines

    for case in tok_data["eval"]["input_ids"]:
        detok_case = happy_gen.tokenizer.decode(case, skip_special_tokens=True)
        assert detok_case in lines


def test_gen_text_decode():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")

    train_data = "../data/gen/train-eval.txt"
    save_path = "data/gen/decode-test/"

    args = GENTrainArgs(num_train_epochs=1,
                                           eval_ratio=0,
                                           save_preprocessed_data=True,
                                           save_preprocessed_data_path=save_path)

    happy_gen.train(train_data, args=args, eval_filepath=train_data)

    tok_data = load_from_disk(save_path)

    with open(train_data, newline='') as text_file:
        file_contents = text_file.read() + "\n"

    for case in tok_data["train"]["input_ids"]:
        detok_case = happy_gen.tokenizer.decode(case)
        assert detok_case == file_contents

    for case in tok_data["eval"]["input_ids"]:
        detok_case = happy_gen.tokenizer.decode(case)
        assert detok_case == file_contents


def test_tt_decode():
    happy_tt = HappyTextToText("T5", "t5-small")

    train_data = "../data/tt/train-eval-grammar.csv"
    save_path = "data/tt/decode-test/"

    args = TTTrainArgs(num_train_epochs=1,
                       save_preprocessed_data=True,
                       save_preprocessed_data_path=save_path,
                       max_input_length=1024,
                       max_output_length=1024)

    happy_tt.train(train_data, args=args)

    tok_data = load_from_disk(save_path)

    inputs = []
    targets = []

    with open(train_data, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            inputs.append(row['input'] + "</s>")
            targets.append(row['target'] + "</s>")

    for case in tok_data["train"]:
        intput_case = happy_tt.tokenizer.decode(case["input_ids"])
        target_case = happy_tt.tokenizer.decode(case["labels"])
        # we shuffle the training cases so we just confirm the detokenized results
        # exist within the raw data
        assert intput_case in inputs
        assert target_case in targets



def test_wp_csv_decode():
    happy_wp = HappyWordPrediction('BERT', 'bert-base-uncased')

    train_data = "../data/wp/train-eval.csv"
    save_path = "data/wp/decode-test/"

    args_max_len_truncation = WPTrainArgs(num_train_epochs=1,
                                           save_preprocessed_data=True,
                                           save_preprocessed_data_path=save_path, padding=False)

    happy_wp.train(train_data, args=args_max_len_truncation)

    tok_data = load_from_disk(save_path)

    lines = []
    with open(train_data, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lines.append(row["text"])

    def clean_detok_case(detok_case):
        detok_case = detok_case.replace("[CLS] line", "Line")
        detok_case = detok_case.replace(" : this", ": This")
        detok_case = detok_case.replace(" [SEP]", "")

        return detok_case
    for case in tok_data["train"]["input_ids"]:
        print(case)
        detok_case = happy_wp.tokenizer.decode(case, skip_special_tokens=False)
        detok_case = clean_detok_case(detok_case)
        assert detok_case in lines

    for case in tok_data["eval"]["input_ids"]:
        detok_case = happy_wp.tokenizer.decode(case, skip_special_tokens=False)
        detok_case = clean_detok_case(detok_case)
        assert detok_case in lines
