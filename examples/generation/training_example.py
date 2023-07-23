import csv
from datasets import load_dataset
from happytransformer import HappyGeneration,  GENTrainArgs


def main():
    train_csv_path = "train.csv"
    eval_csv_path = "eval.csv"

    train_dataset = load_dataset('billsum', split='train[0:1999]')
    eval_dataset = load_dataset('billsum', split='test[0:199]')

    generate_csv(train_csv_path, train_dataset)
    generate_csv(eval_csv_path, eval_dataset)

    happy_gen = HappyGeneration(model_type="GPT2", model_name="gpt2")

    train_args = GENTrainArgs(
        num_train_epochs=1,
        learning_rate=1e-5,
        # report_to = ('wandb'),
        # project_name = "happy-transformer-examples",
        # run_name = "text-generation",
        # deepspeed="ZERO-2"
    )

    happy_gen.train(train_csv_path, args=train_args, eval_filepath=eval_csv_path)

    # happy_gen.push_to_hub("EricFillion/text-generation-example")


def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["text"])

        for case in dataset:
            text = case["summary"]
            writter.writerow([text])


if __name__ == "__main__":
    main()