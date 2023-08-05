import csv
from happytransformer.happy_text_to_text import HappyTextToText, TTTrainArgs
from datasets import load_dataset

def main():
    train_csv_path = "train.csv"
    eval_csv_path = "eval.csv"
    happy_tt = HappyTextToText("MT5", "google/mt5-base")

    train_dataset = load_dataset("persiannlp/parsinlu_translation_en_fa", split='train[0:1999]')
    eval_dataset = load_dataset("persiannlp/parsinlu_translation_en_fa", split='validation[0:199]')

    generate_csv(train_csv_path, train_dataset)
    generate_csv(eval_csv_path, eval_dataset)

    train_args = TTTrainArgs(
                learning_rate=1e-5,
                num_train_epochs=1,
                max_input_length=1024,
                max_output_length=1024,
                # fp16=True,
                # report_to = ('wandb'),
                # project_name = "happy-transformer-examples",
                # run_name = "translation",
                # deepspeed="ZERO-2"
    )

    happy_tt.train(train_csv_path, args=train_args)

    happy_tt.save("finetuned-model/")



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