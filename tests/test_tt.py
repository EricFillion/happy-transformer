from happytransformer import HappyTextToText, TextToTextSettings



def test_default_simple():
    happy_gen = HappyTextToText()
    output = happy_gen.generate_text("translate English to French: Hello my name is Eric")
    assert type(output.text) == str

def test_default_min_max_length():
    happy_gen = HappyTextToText()
    args = TextToTextSettings(min_length=5, max_length=5)
    output = happy_gen.generate_text("translate English to French: Hello my name is Eric", args=args)
    tokens = happy_gen.tokenizer.encode(output.text, return_tensors="pt")
    length = len(tokens[0])
    assert length == 5

def test_all_methods():
    happy_gen = HappyTextToText()

    greedy_settings = TextToTextSettings(min_length=5, max_length=5, no_repeat_ngram_size=2)
    output_greedy = happy_gen.generate_text(
        "translate English to French: Hello my name is Eric",
        args=greedy_settings)


    beam_settings = TextToTextSettings(min_length=5, max_length=5, early_stopping=True, num_beams=5)

    output_beam_search = happy_gen.generate_text(
        "translate English to French: Hello my name is Eric",
        args=beam_settings)

    generic_sampling_settings = TextToTextSettings(min_length=5, max_length=5, do_sample=True, early_stopping=False, top_k=0, temperature=0.7)

    output_generic_sampling = happy_gen.generate_text(
        "translate English to French: Hello my name is Eric",
        args=generic_sampling_settings)

    top_k_sampling_settings = TextToTextSettings(min_length=5, max_length=5, do_sample=True, early_stopping=False, top_k=50, temperature=0.7)

    output_top_k_sampling = happy_gen.generate_text(
        "translate English to French: Hello my name is Eric",
        args=top_k_sampling_settings)

    top_p_sampling_settings = TextToTextSettings(min_length=5, max_length=5, do_sample=True, early_stopping=False, top_k=0, top_p=0.8, temperature=0.7)

    output_top_p_sampling = happy_gen.generate_text(
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
    happy_tt = HappyTextToText()
    happy_tt.save("model/")
    result_before = happy_tt.generate_text("translate English to French: Hello my name is Eric")

    happy = HappyTextToText(load_path="model/")
    result_after = happy.generate_text("translate English to French: Hello my name is Eric")

    assert result_before.text == result_after.text