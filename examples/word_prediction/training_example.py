import csv
from datasets import load_dataset
from happytransformer.happy_word_prediction import HappyWordPrediction, WPEvalArgs, WPTrainArgs


def main():
    train_csv_path = "train.csv"
    eval_csv_path = "eval.csv"

    train_dataset = load_dataset('billsum', split='train[0:1999]')
    eval_dataset = load_dataset('billsum', split='test[0:499]')

    generate_csv(train_csv_path, train_dataset)
    generate_csv(eval_csv_path, eval_dataset)

    happy_wp = HappyWordPrediction(model_type="DISTILBERT", model_name="distilbert-base-uncased")

    train_args = WPTrainArgs(
        learning_rate=1e-5,
        num_train_epochs=1,
        # report_to = ('wandb'),
        # project_name = "happy-transformer-examples",
        # run_name = "word-prediction",
        # deepspeed="ZERO-2"
    )

    happy_wp.train(train_csv_path, args=train_args, eval_filepath=eval_csv_path)

    # happy_wp.push_to_hub("EricFillion/wp-example")


def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["text"])
        for case in dataset:
            text = case["summary"]
            writter.writerow([text])


if __name__ == "__main__":
    main()