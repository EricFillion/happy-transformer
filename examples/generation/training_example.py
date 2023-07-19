import csv
from datasets import load_dataset
from happytransformer import HappyGeneration, GENTrainArgs


def main():
    train_csv_path = "train.csv"
    eval_csv_path = "eval.csv"

    train_dataset = load_dataset('billsum', split='train[0:1999]')
    eval_dataset = load_dataset('billsum', split='test[0:1999]')

    generate_csv(train_csv_path, train_dataset)
    generate_csv(eval_csv_path, eval_dataset)

    happy_gen = HappyGeneration(model_type="GPT2", model_name="gpt2")
    starter_texts = ["Authorizes", "Allows", "Prevents", "Restricts"]

    print("Examples Before Training: ")
    produce_examples(starter_texts, happy_gen)

    before_loss = happy_gen.eval(eval_csv_path)

    # uncomment the deepspeed parameter to use Deepspeed
    args = GENTrainArgs(
        padding=True,
        # deepspeed="../deepspeed/ds_config.json",
        # report_to = tuple(['wandb'])
        save_preprocessed_data=True,
        save_preprocessed_data_path="/tok-data/",
        # enable the code below to reused preprocesses data for your second run
        # load_preprocessed_data=True,
        # save_preprocessed_data_path="/tok-data/"
    )

    happy_gen.train(train_csv_path, args=args, eval_filepath=eval_csv_path)

    after_loss = happy_gen.eval(eval_csv_path)

    print("Before loss: ", before_loss.loss)
    print("After loss: ", after_loss.loss, end="\n\n")

    print("Examples After Training: ")
    produce_examples(starter_texts, happy_gen)


def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["text"])

        for case in dataset:
            text = case["summary"]
            writter.writerow([text])

def produce_examples(starter_texts, happy_gen):
    for start in starter_texts:
        output = happy_gen.generate_text(start)
        print(start, output.text)


if __name__ == "__main__":
    main()