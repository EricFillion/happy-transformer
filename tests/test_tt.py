import torch
from happytransformer import HappyTextToText, TTSettings, TTTrainArgs, TTEvalArgs
from tests import happy_tt
from tests.run_save_load import run_save_load

def test_default_simple():
    output = happy_tt.generate_text("translate English to French: Hello my name is Eric")
    assert type(output.text) == str

def test_default_min_max_length():
    args = TTSettings(min_length=5, max_length=5)
    output = happy_tt.generate_text("translate English to French: Hello my name is Eric", args=args)
    tokens = happy_tt.tokenizer.encode(output.text, return_tensors="pt")
    length = len(tokens[0])
    assert length == 5

def test_all_methods():
    greedy_settings = TTSettings(min_length=5, max_length=5, no_repeat_ngram_size=2)
    output_greedy = happy_tt.generate_text(
        "translate English to French: Hello my name is Eric",
        args=greedy_settings)


    beam_settings = TTSettings(min_length=5, max_length=5, early_stopping=True, num_beams=5)

    output_beam_search = happy_tt.generate_text(
        "translate English to French: Hello my name is Eric",
        args=beam_settings)

    generic_sampling_settings = TTSettings(min_length=5, max_length=5, do_sample=True, early_stopping=False, top_k=0, temperature=0.7)

    output_generic_sampling = happy_tt.generate_text(
        "translate English to French: Hello my name is Eric",
        args=generic_sampling_settings)

    top_k_sampling_settings = TTSettings(min_length=5, max_length=5, do_sample=True, early_stopping=False, top_k=50, temperature=0.7)

    output_top_k_sampling = happy_tt.generate_text(
        "translate English to French: Hello my name is Eric",
        args=top_k_sampling_settings)

    top_p_sampling_settings = TTSettings(min_length=5, max_length=5, do_sample=True, early_stopping=False, top_k=0, top_p=0.8, temperature=0.7)

    output_top_p_sampling = happy_tt.generate_text(
        "translate English to French: Hello my name is Eric",
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

def test_tt_save():
    happy_tt.save("model/")
    result_before = happy_tt.generate_text("translate English to French: Hello my name is Eric")

    happy = HappyTextToText(model_name="model/")
    result_after = happy.generate_text("translate English to French: Hello my name is Eric")

    assert result_before.text == result_after.text


def test_tt_train_simple():
    happy_tt.train("../data/tt/train-eval-grammar.csv")

def test_tt_train_fp16():
    # Can only be used if fp16 if CUDA is available
    if torch.cuda.is_available():
        args = TTTrainArgs(fp16=True)
        happy_tt.train("../data/tt/train-eval-grammar.csv", args=args)

def test_tt_eval_simple():
    happy_tt.eval("../data/tt/train-eval-grammar.csv")

def test_tt_eval_loss_decreases():
    happy = HappyTextToText("T5", "t5-small")
    args = TTTrainArgs(num_train_epochs=3)
    before_result = happy.eval("../data/tt/train-eval-grammar.csv")
    happy.train("../data/tt/train-eval-grammar.csv", args=args)
    after_result = happy_tt.eval("../data/tt/train-eval-grammar.csv")

    assert before_result.loss > after_result.loss

def test_tt_train_custom_p():
    args = TTTrainArgs(num_train_epochs=2, max_input_length=100, max_output_length=100, batch_size=2)
    happy_tt.train("../data/tt/train-eval-grammar.csv", args=args)

def test_tt_eval_custom_p():
    args = TTEvalArgs(max_input_length=100, max_output_length=100, batch_size=2)
    result = happy_tt.eval("../data/tt/train-eval-grammar.csv", args=args)
    assert type(result.loss) is float


def test_tt_save_load_train():
    output_path = "data/tt-train/"
    data_path = "../data/tt/train-eval-grammar.csv"
    args = TTTrainArgs(num_train_epochs=1)

    run_save_load(happy_tt, output_path, args, data_path, "train")

def test_tt_save_load_eval():
    output_path = "data/tt-train/"
    data_path = "../data/tt/train-eval-grammar.csv"

    args = TTTrainArgs(num_train_epochs=1)
    run_save_load(happy_tt, output_path, args, data_path, "eval")
