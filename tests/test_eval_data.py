from happytransformer import (
    HappyGeneration,
    GENTrainArgs)
import pytest
from tests import happy_gen

DATA_PATH = "../data/gen/train-eval.txt"

# tests what happens when no eval file is provided and eval_ratio is set to 0.
def test_gen_eval_0():

    args = GENTrainArgs(eval_ratio=0.0, num_train_epochs=1)

    with pytest.raises(ValueError):
        happy_gen.train(DATA_PATH, args=args)

def test_gen_eval_ratio():

    args = GENTrainArgs(eval_ratio=0.1, num_train_epochs=1)

    happy_gen.train(DATA_PATH, args=args)


def test_gen_eval_path():

    args = GENTrainArgs(eval_ratio=0.1, num_train_epochs=1)

    happy_gen.train(DATA_PATH, args=args, eval_filepath=DATA_PATH)



def test_gen_saved_data():
    save_path = "data/test/"
    args_save = GENTrainArgs(eval_ratio=0.1, num_train_epochs=1, save_path=save_path)

    happy_gen.train(DATA_PATH, args=args_save)

    args_load = GENTrainArgs(num_train_epochs=1, load_path=save_path)

    happy_gen.train(DATA_PATH, args=args_load)

    # Should still work even if eval_filepath is provided
    happy_gen.train(DATA_PATH, eval_filepath=DATA_PATH,  args=args_load)





