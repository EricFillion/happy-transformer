from happytransformer.happy_trainer import HappyTrainer
from happytransformer.mwp.default_args import ARGS_MWP_TRAIN


class WPTrainer(HappyTrainer):

    def train(self, input_filepath, args=ARGS_MWP_TRAIN):
        raise NotImplementedError()

    def eval(self, input_filepath):
        raise NotImplementedError()

    def test(self, input_filepath, pipeline):
        raise NotImplementedError()