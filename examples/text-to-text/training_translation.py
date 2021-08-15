import csv
from happytransformer.happy_text_to_text import HappyTextToText, TTTrainArgs
from datasets import load_dataset

def main():
    happy_tt = HappyTextToText("MT5", "google/mt5-base")
    text = "Hello, I like to eat apples."
    # Google translate translation: سلام من عاشق خوردن سیب هستم.
    before_text = happy_tt.generate_text("translate English to Persian: " + text)

    train_dataset = load_dataset("persiannlp/parsinlu_translation_en_fa", split='train[0:3999]')
    eval_dataset = load_dataset("persiannlp/parsinlu_translation_en_fa", split='validation[0:399]')

    generate_csv("train.csv", train_dataset)
    generate_csv("eval.csv", eval_dataset)

    eval_args = TTTrainArgs(max_input_length=1024, max_output_length=1024)
    before_loss = happy_tt.eval("eval.csv", args=eval_args)

    train_args = TTTrainArgs(num_train_epochs=1, max_input_length=1024, max_output_length=1024)
    happy_tt.train("train.csv", args=train_args)

    after_text = happy_tt.generate_text("translate English to Persian: " + text)

    after_loss = happy_tt.eval("eval.csv", args=eval_args)

    print("before loss:", before_loss)
    print("after loss:", after_loss)

    print("before text:", before_text)
    print("after text:", after_text)


def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["input", "target"])
        for case in dataset:
            english_text = "translate English to Persian: " + case["source"]
            persian_text = case["targets"][0]
            writter.writerow([english_text, persian_text])


if __name__ == "__main__":
    main()