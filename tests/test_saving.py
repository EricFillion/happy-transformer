from happytransformer import HappyTextClassification, TCTrainArgs
from tests import happy_tc


def test_logging_and_saving():

    args = TCTrainArgs(save_steps=0.5, num_train_epochs=3)

    happy_tc.train("../data/tc/train-eval.csv", args=args)

    happy = HappyTextClassification(model_type="BERT",
                                       model_name="happy_transformer/checkpoint-5")