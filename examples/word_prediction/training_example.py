import csv
from datasets import load_dataset
from happytransformer.happy_word_prediction import HappyWordPrediction, WPTrainArgs


def main():
    train_txt_path = "train.txt"
    eval_txt_path = "eval.txt"

    train_dataset = load_dataset('billsum', split='train[0:1999]')
    eval_dataset = load_dataset('billsum', split='test[0:199]')

    generate_txt_file(train_txt_path, train_dataset)
    generate_txt_file(eval_txt_path, eval_dataset)

    happy_wp = HappyWordPrediction(model_type="DISTILBERT", model_name="distilbert-base-uncased")

    train_args = WPTrainArgs(
        learning_rate=1e-5,
        num_train_epochs=1,
        # fp16=True,
        # report_to = ('wandb'),
        # project_name = "happy-transformer-examples",
        # run_name = "word-prediction",
        # deepspeed="ZERO-2"
    )

    happy_wp.train(train_txt_path, args=train_args, eval_filepath=eval_txt_path)

    happy_wp.save("finetuned-model/")


def generate_txt_file(csv_path, dataset):
    with open(csv_path, 'w', newline='') as text_fule:
        for case in dataset:
            text = case["summary"]
            text_fule.write(text + "\n")



if __name__ == "__main__":
    main()