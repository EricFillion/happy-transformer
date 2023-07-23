from datasets import load_dataset
import csv
from happytransformer import  HappyQuestionAnswering, QATrainArgs


def main():
    # Be careful not to commit the csv files to the rep
    train_csv_path = "train.csv"
    eval_csv_path = "eval.csv"

    train_dataset = load_dataset('squad', split='train[0:499]')
    eval_dataset = load_dataset('squad', split='validation[0:99]')

    generate_csv(train_csv_path, train_dataset)
    generate_csv(eval_csv_path, eval_dataset)

    happy_qa = HappyQuestionAnswering(model_type="BERT", model_name="bert-base-uncased")


    train_args = QATrainArgs(
        # report_to = ('wandb'),
        # project_name = "happy-transformer-examples",
        # run_name = "question-answering",
        # deepspeed="ZERO-2",

    )
    happy_qa.train(train_csv_path, args=train_args, eval_filepath=eval_csv_path)

    # happy_qa.push_to_hub("EricFillion/question-answering-example)


def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["context", "question", "answer_text", "answer_start"])
        for case in dataset:
            context = case["context"]
            question = case["question"]
            answer_text = case["answers"]["text"][0]
            answer_start = case["answers"]["answer_start"][0]
            writter.writerow([context, question, answer_text, answer_start])


if __name__ == "__main__":
    main()
