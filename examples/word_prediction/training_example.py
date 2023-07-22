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
    eval_args = WPEvalArgs(
        #deepspeed=True
    )
    starter_texts = ["Authorizes [MASK] for community use",
                     "Allows for [MASK] to be used at gatherings of over 50 people.",
                     "Prevents children under 18 years old from buying [MASK]",
                     "Restricts [MASK] from being sold"]

    print("Examples Before Training: ")
    produce_examples(starter_texts, happy_wp)

    before_loss = happy_wp.eval(eval_csv_path, args=eval_args)

    train_args = WPTrainArgs(
        #deepspeed=True,
        # report_to = tuple(['wandb'])

    )
    happy_wp.train(train_csv_path, args=train_args, eval_filepath=eval_csv_path)

    after_loss = happy_wp.eval(eval_csv_path, args=eval_args)

    print("Before loss: ", before_loss.loss)
    print("After loss: ", after_loss.loss, end="\n\n")

    print("Examples After Training: ")
    produce_examples(starter_texts, happy_wp)


def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["text"])
        for case in dataset:
            text = case["summary"]
            writter.writerow([text])


def produce_examples(starter_texts, happy_wp):
    for start in starter_texts:
        output = happy_wp.predict_mask(start)
        print(start, "Output: ", output[0].token)


if __name__ == "__main__":
    main()