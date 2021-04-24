from datasets import load_dataset
from happytransformer.happy_generation import HappyGeneration


def main():
    train_csv_path = "train.txt"
    eval_csv_path = "eval.txt"

    train_dataset = load_dataset('billsum', split='train[0:1999]')
    eval_dataset = load_dataset('billsum', split='test[0:499]')

    generate_txt(train_csv_path, train_dataset)
    generate_txt(eval_csv_path, eval_dataset)

    happy_gen = HappyGeneration(model_type="GPT2", model_name="gpt2")
    starter_texts = ["Authorizes", "Allows", "Prevents", "Restricts"]

    print("Examples Before Training: ")
    produce_examples(starter_texts, happy_gen)

    before_loss = happy_gen.eval(eval_csv_path)

    happy_gen.train(train_csv_path)

    after_loss = happy_gen.eval(eval_csv_path)

    print("Before loss: ", before_loss.loss)
    print("After loss: ", after_loss.loss, end="\n\n")

    print("Examples After Training: ")
    produce_examples(starter_texts, happy_gen)


def generate_txt(csv_path, dataset):
    with open(csv_path, 'w', newline='') as text_fule:
        for case in dataset:
            text = case["summary"]
            text_fule.write(text + "\n")

def produce_examples(starter_texts, happy_gen):
    for start in starter_texts:
        output = happy_gen.generate_text(start)
        print(start, output.text)


if __name__ == "__main__":
    main()