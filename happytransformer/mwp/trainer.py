from happytransformer.trainer import Trainer
from happytransformer.mwp.default_args import ARGS_MWP_EVAL, ARGS_MWP_TEST, ARGS_MWP_TRAIN


class QATrainer(Trainer):

    def __init__(self, model, model_type, tokenizer, device, logger):
        super(QATrainer, self).__init__(model, model_type, tokenizer, device, logger)

    def train(self, input_filepath, args=ARGS_MWP_TRAIN):
        raise NotImplementedError()

    def test(self, input_filepath, answers_to_question, output_filepath, args=ARGS_MWP_TEST):
        raise NotImplementedError()

    def eval(self, input_filepath, answers_to_question, args=ARGS_MWP_EVAL, output_filepath=None):
        raise NotImplementedError()