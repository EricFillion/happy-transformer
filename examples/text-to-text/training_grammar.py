import csv
from happytransformer.happy_text_to_text import HappyTextToText, TTTrainArgs
from datasets import load_dataset

def main():
    happy_tt = HappyTextToText("T5", "t5-base")
    input_text = "grammar: This sentences had bad grammars and spelling. "
    before_text = happy_tt.generate_text(input_text).text
    train_csv_path = "train.csv"
    eval_csv_path = "eval.csv"

    # There's no training split. Just eval and test. So, we'll use eval for train and test for eval.
    # 755 cases, but each case has 4 corrections so there are really 3020
    train_dataset = load_dataset("jfleg", split='validation[:]')

    # 748 cases, but again, each case has 4 correction so there are really
    eval_dataset = load_dataset("jfleg", split='test[:]')

    generate_csv("train.csv", train_dataset)
    generate_csv("eval.csv", eval_dataset)

    before_loss = happy_tt.eval(eval_csv_path, args=eval_args).loss
    train_args = TTTrainArgs(
                        # deepspeed=True,
                        # report_to = tuple(['wandb'])
    )
    happy_tt.train(train_csv_path, args=train_args, eval_filepath=eval_csv_path)

    after_text = happy_tt.generate_text(input_text).text

    print("------------------------------------")

    print("input text:", input_text)
    print("before text:", before_text)
    print("after text:", after_text)


def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["input", "target"])
        for case in dataset:
            input_text = "grammar: " + case["sentence"]
            for correction in case["corrections"]:
                # a few of the case are have None values. We'll skip them
                if input_text and correction:
                    writter.writerow([input_text, correction])

if __name__ == "__main__":
    main()
