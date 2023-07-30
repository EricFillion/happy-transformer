import csv
from happytransformer.happy_text_to_text import HappyTextToText, TTTrainArgs
from datasets import load_dataset

def main():
    happy_tt = HappyTextToText("T5", "t5-base")
    train_csv_path = "train.csv"
    eval_csv_path = "eval.csv"

    train_dataset = load_dataset("xsum", split='train[0:1999]')
    eval_dataset = load_dataset("xsum", split='validation[0:199]')

    generate_csv(train_csv_path, train_dataset)
    generate_csv(eval_csv_path, eval_dataset)

    train_args = TTTrainArgs(
                       num_train_epochs=1,
                       learning_rate=1e-5,
                       max_input_length=1024,
                       max_output_length=128,
                       # report_to = ('wandb'),
                       # project_name = "happy-transformer-examples",
                       # run_name = "summarization",
                       # deepspeed="ZERO-2"
                    )

    happy_tt.train(train_csv_path, args=train_args)

    happy_tt.save("finetuned-model/")

def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["input", "target"])
        for case in dataset:
            long_text = "summarize" + case["document"]
            short_text = case["summary"]
            writter.writerow([long_text, short_text])



if __name__ == "__main__":
    main()