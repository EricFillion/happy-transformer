from datasets import load_dataset
import csv
from happytransformer import  HappyTextClassification, TCEvalArgs, TCTrainArgs


def main():
    train_csv_path = "train.csv"
    eval_csv_path = "eval.csv"

    train_dataset = load_dataset('go_emotions', split='train[0:1999]')
    eval_dataset = load_dataset('go_emotions', split='validation[0:399]')

    generate_csv(train_csv_path, train_dataset)
    generate_csv(eval_csv_path, eval_dataset)

    happy_tc = HappyTextClassification(model_type="BERT", model_name="bert-base-uncased", num_labels=28)

    train_args = TCTrainArgs(
                       #deepspeed=True,
                       # report_to = tuple(['wandb'])
                       )

    happy_tc.train(train_csv_path, args=train_args, eval_filepath=eval_csv_path)



def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["text", "label"])
        for case in dataset:
            # some cases have multiple labels,
            # so each one becomes its own training case
            for label in case["labels"]:
                text = case["text"]
                writter.writerow([text, label])


if __name__ == "__main__":
    main()