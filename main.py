from happy_transformer.happy_xlnet import HappyXLNET


def main():
    """testing"""

    xl = HappyXLNET(model="xlnet-base-cased")

    xl.init_sequence_classifier()
    xl.train_sequence_classifier(csv_path="data/train.csv")

    #results = xl.eval_sequence_classifier(csv_path="data/eval.csv")

    #print("Eval results", results)
    results = xl.test_sequence_classifier(csv_path="data/test.csv")
    print("Test results", results)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Aborting')
