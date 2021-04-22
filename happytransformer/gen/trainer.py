from happytransformer.happy_trainer import HappyTrainer
from happytransformer.gen.default_args import ARGS_GEN_TRAIN


class GENTrainer(HappyTrainer):

    def train(self, input_filepath, args=ARGS_GEN_TRAIN):
        raise NotImplementedError()

    def eval(self, input_filepath):
        raise NotImplementedError()

    def test(self, input_filepath, pipeline):
        raise NotImplementedError()