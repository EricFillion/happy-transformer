import torch

from happytransformer import (
    HappyGeneration,
    GENSettings,
    GENTrainArgs,
    GENEvalArgs
)
import pytest

from tests.run_save_load import run_save_load

def test_default_simple():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    args = GENSettings(min_length=5, max_length=5)
    output = happy_gen.generate_text("Artificial intelligence is ", args=args)
    assert type(output.text) == str


def test_default_min_max_length():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    args = GENSettings(min_length=5, max_length=5)
    output = happy_gen.generate_text("Artificial intelligence is ", args=args)
    tokens = happy_gen.tokenizer.encode(output.text, return_tensors="pt")
    length = len(tokens[0])
    assert length == 5

def test_bad_words():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    # Test single words
    args_test_word_single = GENSettings(bad_words=["new", "tool"])
    # result without bad_words:  "a new field of research that has been gaining momentum"
    output_single = happy_gen.generate_text("Artificial intelligence is ", args=args_test_word_single)
    output_words_single = output_single.text.split()
    for phrase in args_test_word_single.bad_words:
        for word in phrase.split():
            assert word not in output_words_single
    # Test phrases 
    args_test_word_phrase = GENSettings(bad_words=["new field"])
    output_phrase = happy_gen.generate_text("Artificial intelligence is ", args=args_test_word_phrase)
    for phrase in args_test_word_phrase.bad_words:
        assert phrase not in output_phrase.text


def test_top_p():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    # Test small values
    args_small = GENSettings(top_p=0.01,  min_length=20, max_length=20)
    output_small = happy_gen.generate_text("Artificial intelligence is ", args=args_small)
    tokens_small = happy_gen.tokenizer.encode(output_small.text, return_tensors="pt")
    length_small = len(tokens_small[0])
    assert length_small == 20

    # Test large values
    args_large = GENSettings(top_p=1, min_length=20, max_length=20)
    output_large = happy_gen.generate_text("Artificial intelligence is ", args=args_large)
    tokens_large = happy_gen.tokenizer.encode(output_large.text, return_tensors="pt")
    length_large = len(tokens_large[0])
    assert length_large == 20



def test_all_methods():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")

    greedy_settings = GENSettings(min_length=5, max_length=5, no_repeat_ngram_size=2)
    output_greedy = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=greedy_settings)


    beam_settings = GENSettings(min_length=5, max_length=5, early_stopping=True, num_beams=5)

    output_beam_search = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=beam_settings)

    generic_sampling_settings = GENSettings(min_length=5, max_length=5, do_sample=True, early_stopping=False, top_k=0, temperature=0.7)

    output_generic_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=generic_sampling_settings)

    top_k_sampling_settings = GENSettings(min_length=5, max_length=5, do_sample=True, early_stopping=False, top_k=50, temperature=0.7)

    output_top_k_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=top_k_sampling_settings)

    top_p_sampling_settings = GENSettings(min_length=5, max_length=5, do_sample=True, early_stopping=False, top_k=0, top_p=0.8, temperature=0.7)

    output_top_p_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=top_p_sampling_settings)

    assert type(output_greedy.text) == str
    assert type(output_beam_search.text) == str
    assert type(output_generic_sampling.text) == str
    assert type(output_top_k_sampling.text) == str
    assert type(output_top_p_sampling.text) == str

    print("greedy: ", output_greedy.text, end="\n\n")
    print("beam-search: ", output_beam_search.text, end="\n\n")
    print("generic-sampling: ", output_generic_sampling.text, end="\n\n")
    print("top-k-sampling: ", output_top_k_sampling.text, end="\n\n")
    print("top-p-sampling: ", output_top_p_sampling.text, end="\n\n")

def test_gen_train_basic():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    happy_gen.train("../data/gen/train-eval.txt")

def test_gen_train_hp():
    # Can only be used if fp16 if CUDA is available
    if torch.cuda.is_available():
        happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
        args = GENTrainArgs(fp16=True)
        happy_gen.train("../data/gen/train-eval.txt", args=args)

def test_gen_eval_basic():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    result = happy_gen.eval("../data/gen/train-eval.txt")
    assert type(result.loss) == float

def test_gen_train_effectiveness_multi():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    before_result = happy_gen.eval("../data/gen/train-eval.txt")
    happy_gen.train("../data/gen/train-eval.txt")
    after_result = happy_gen.eval("../data/gen/train-eval.txt")

    assert after_result.loss < before_result.loss

def test_gen_save_load_train():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    output_path = "data/gen-train/"
    data_path = "../data/gen/train-eval.txt"
    args = GENTrainArgs()
    run_save_load(happy_gen, output_path, args, data_path, "train")

def test_gen_save_load_eval():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    output_path = "data/gen-eval/eval/"
    data_path = "../data/gen/train-eval.txt"
    args = GENEvalArgs()

    run_save_load(happy_gen, output_path, args, data_path, "eval")

def test_gen_save():
    happy = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    happy.save("model/")
    result_before = happy.generate_text("Natural language processing is")

    happy = HappyGeneration(model_name="model/")
    result_after = happy.generate_text("Natural language processing is")

    assert result_before == result_after

def test_wp_train_eval_with_dic():

    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    train_args = {'learning_rate': 0.01,  "num_train_epochs": 1}

    with pytest.raises(ValueError):
        happy_gen.train("../data/gen/train-eval.txt" , args=train_args)
    eval_args = {}

    with pytest.raises(ValueError):
        result = happy_gen.eval("../data/gen/train-eval.txt", args=eval_args)


def test_gen_train_eval_with_dataclass():

    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    train_args = GENTrainArgs(learning_rate=0.01, num_train_epochs=1)

    happy_gen.train("../data/gen/train-eval.txt" , args=train_args)

    eval_args = GENEvalArgs()

    after_result = happy_gen.eval("../data/wp/train-eval.txt", args=eval_args)

    assert type(after_result.loss) == float

def test_generate_after_train_eval():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    happy_gen.train("../data/gen/train-eval.txt")
    eval_result = happy_gen.eval("../data/gen/train-eval.txt")
    output = happy_gen.generate_text("Artificial intelligence is ")
    assert type(output.text) == str

def test_gen_csv():
    data_path = "../data/gen/train-eval.csv"
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    before_result = happy_gen.eval(data_path)
    print("before_result", before_result)
    happy_gen.train(data_path)
    after_result = happy_gen.eval(data_path)
    print("after_result", after_result)

    assert after_result.loss < before_result.loss
