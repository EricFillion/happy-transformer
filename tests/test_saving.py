from happytransformer import HappyTextClassification, TCTrainArgs

def test_logging_and_saving():
    happy_tc = HappyTextClassification(
        model_type="BERT",
        model_name="prajjwal1/bert-tiny"
    )
    args = TCTrainArgs(save_steps=0.5)

    happy_tc.train("../data/tc/train-eval.csv", args=args)

    happy_tc = HappyTextClassification(model_type="BERT",
                                       model_name="prajjwal1/bert-tiny",
                                       load_path = "happy_transformer/checkpoint-5")