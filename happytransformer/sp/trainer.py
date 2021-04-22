from happytransformer.happy_trainer import HappyTrainer
from happytransformer.sp.default_args import ARGS_SP_TRAIN


class SPTrainer(HappyTrainer):

    def train(self, input_filepath, args=ARGS_SP_TRAIN):
        raise NotImplementedError()

    def eval(self, input_filepath, args):
        raise NotImplementedError()

    def test(self, input_filepath, solve, args):
        raise NotImplementedError()