from happytransformer.happy_trainer import HappyTrainer
from happytransformer.toc.default_args import ARGS_TOC_TRAIN


class TOCTrainer(HappyTrainer):

    def train(self, input_filepath, args=ARGS_TOC_TRAIN):
        raise NotImplementedError()

    def eval(self, input_filepath, args):
        raise NotImplementedError()

    def test(self, input_filepath, solve, args):
        raise NotImplemented