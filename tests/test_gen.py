from happytransformer import HappyGeneration, ARGS_GEN_TRAIN, ARGS_GEN_EVAl
from happytransformer import(
    GEN_DEFAULT_SETTINGS,
    GEN_GREEDY_SETTINGS,
    GEN_BEAM_SETTINGS,
    GEN_TOP_K_SAMPLING_SETTINGS,
    GEN_GENERIC_SAMPLING_SETTINGS
)
from tests.shared_tests import run_save_load_train


def test_default_simple():
    happy_gen = HappyGeneration()
    output = happy_gen.generate_text("Artificial intelligence is ", min_length=5, max_length=5)
    assert type(output.text) == str
    print("default simple: ", output.text)


def test_default_min_max_length():
    happy_gen = HappyGeneration()
    output = happy_gen.generate_text("Artificial intelligence is ", min_length=5, max_length=5)
    tokens = happy_gen.tokenizer.encode(output.text, return_tensors="pt")
    length = len(tokens[0])
    assert length == 5


def test_all_methods():
    happy_gen = HappyGeneration()
    output_greedy = happy_gen.generate_text(
        "Artificial intelligence is ",
        settings=GEN_GREEDY_SETTINGS, min_length=5, max_length=5)

    output_beam_search = happy_gen.generate_text(
        "Artificial intelligence is ",
        settings=GEN_BEAM_SETTINGS, min_length=5, max_length=5)

    output_generic_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        settings=GEN_GENERIC_SAMPLING_SETTINGS, min_length=5, max_length=5)

    output_top_k_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        settings=GEN_TOP_K_SAMPLING_SETTINGS, min_length=5, max_length=5)

    assert type(output_greedy.text) == str
    assert type(output_beam_search.text) == str
    assert type(output_generic_sampling.text) == str
    assert type(output_top_k_sampling.text) == str

    print("greedy: ", output_greedy.text, end="\n\n")
    print("beam-search: ", output_beam_search.text, end="\n\n")
    print("generic-sampling: ", output_generic_sampling.text, end="\n\n")
    print("top-k-sampling: ", output_top_k_sampling.text, end="\n\n")

def test_full_settings():
    happy_gen = HappyGeneration()
    output = happy_gen.generate_text(
        "Artificial intelligence is ",
        settings=GEN_DEFAULT_SETTINGS, min_length=5, max_length=20)

    assert type(output.text) == str
    print("Full settings: ", output.text, end="\n\n")


def test_outside_settings():
    """
    Changes settings that we did not include within full_settings.
    Only advanced users will be interested in this feature.

    Uses settings for Top-p Nucleus Sampling

    """
    settings = {
        "do_sample": True,
        'max_length': 50,
        'top_k': 50,
        'top_p': 0.95
    }
    happy_gen = HappyGeneration()
    output = happy_gen.generate_text(
        "Artificial intelligence is ",
        settings=settings, min_length=5, max_length=5)

    assert type(output.text) == str
    print("Outside settings: ", output.text, end="\n\n")



def test_invalid_settings():
    happy_gen = HappyGeneration()
    settings = {
        "age": 22,
        "gender": "male"
    }

    output = happy_gen.generate_text("Artificial intelligence is ",
                                     settings=settings, min_length=5, max_length=5)

    assert type(output.text) == str
    print("invalid settings: ", output.text, end="\n\n")


def test_min_setting_included():
    happy_gen = HappyGeneration()
    settings = {
        "min_length": 22,
    }

    output = happy_gen.generate_text("Artificial intelligence is ",
                                     settings=settings, min_length=5, max_length=5)

    assert type(output.text) == str
    print("Min settings included: ", output.text, end="\n\n")


def test_max_setting_included():
    happy_gen = HappyGeneration()
    settings = {
        "max_length": 22,
    }

    output = happy_gen.generate_text("Artificial intelligence is ",
                                     settings=settings, min_length=5, max_length=5)

    assert type(output.text) == str
    print("Max settings included: ", output.text, end="\n\n")


def test_gen_train_basic():
    happy_gen = HappyGeneration()
    happy_gen.train("../data/gen/train.txt")

def test_gen_eval_basic():
    happy_gen = HappyGeneration()
    result = happy_gen.eval("../data/gen/train.txt")
    assert type(result.loss) == float

def test_gen_train_effectiveness_multi():
    happy_gen = HappyGeneration()
    before_result = happy_gen.eval("../data/gen/train.txt")
    happy_gen.train("../data/gen/train.txt")
    after_result = happy_gen.eval("../data/gen/train.txt")

    assert after_result.loss < before_result.loss

def test_gen_save_load_train():
    happy_gen = HappyGeneration()
    output_path = "data/gen-train.txt"
    data_path = "../data/gen/train.txt"
    run_save_load_train(happy_gen, output_path, ARGS_GEN_TRAIN, data_path, "train")

def test_gen_save_load_eval():
    happy_gen = HappyGeneration()
    output_path = "data/wp-eval.txt"
    data_path = "../data/gen/train.txt"
    run_save_load_train(happy_gen, output_path, ARGS_GEN_EVAl, data_path, "eval")

def test_gen_save():
    happy = HappyGeneration()
    happy.save("model/")
    result_before = happy.generate_text("Natural language processing is")

    happy = HappyGeneration(load_path="model/")
    result_after=happy.generate_text("Natural language processing is")

    assert result_before == result_after
