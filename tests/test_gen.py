from happytransformer import HappyGeneration
from happytransformer import gen_greedy_settings, gen_p_nucleus_sampling_settings, gen_beam_settings, gen_top_k_sampling_settings, gen_generic_sampling_settings

def test_gen_simple():
    happy_gen = HappyGeneration()
    output = happy_gen.generate_text("I went to the basement and then", min_length=5, max_length=40)
    assert type(output.text) == str



def test_gen_min_max_length():
    happy_gen = HappyGeneration()
    output = happy_gen.generate_text("I went to the basement and then",  min_length=5, max_length=5)
    tokens = happy_gen.tokenizer.encode(output.text, return_tensors="pt")
    length = len(tokens[0])
    assert length == 5


def test_all_methods():
    happy_gen = HappyGeneration()
    output_greedy = happy_gen.generate_text("I went to the basement and then", settings=gen_greedy_settings, min_length=5, max_length=20)
    output_beam_search = happy_gen.generate_text("I went to the basement and then", settings=gen_p_nucleus_sampling_settings, min_length=5, max_length=20)
    output_generic_sampling = happy_gen.generate_text("I went to the basement and then", settings=gen_beam_settings, min_length=5, max_length=20)
    output_top_k_sampling = happy_gen.generate_text("I went to the basement and then", settings=gen_top_k_sampling_settings, min_length=5, max_length=20)
    output_top_p_nucleus_sampling= happy_gen.generate_text("I went to the basement and then", settings=gen_generic_sampling_settings, min_length=5, max_length=20)

    print("greedy: ", output_greedy, end="\n\n")
    print("beam-search: ", output_beam_search, end="\n\n")
    print("generic-sampling: ", output_generic_sampling, end="\n\n")
    print("top-k-sampling: ", output_top_k_sampling, end="\n\n")
    print("top-p-nucleus-sampling: ", output_top_p_nucleus_sampling, end="\n\n")

def test_incomplete_settings():
    happy_gen = HappyGeneration()
    settings = {
        "do_sample": False,
        "early_stopping": False,
        "num_beams": 1,
        "temperature": 0.85,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1,
        # "length_penalty": 1,
        # "no_repeat_ngram_size": 2,
        # 'bad_words_ids': None,
    }

    output = happy_gen.generate_text("I went to the basement and then", settings=settings, min_length=5, max_length=20)

    print("custom: ", output)

def test_extra_settings():
    happy_gen = HappyGeneration()
    settings = {
        "do_sample": False,
        "early_stopping": False,
        "num_beams": 1,
        "temperature": 0.85,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1,
        "length_penalty": 1,
        "no_repeat_ngram_size": 2,
        'bad_words_ids': None,
        "age": 22,
        "gender": "male"
    }

    output = happy_gen.generate_text("I went to the basement and then", settings=settings, min_length=5, max_length=20)

    print("custom: ", output)
