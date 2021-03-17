from happytransformer import HappyBackgroundGeneration, HappyGeneration, gen_beam_settings
import spacy

def test_bg_gen_simple():
    happy_gen = HappyGeneration("GPT2", "gpt2")
    happy_bg_gen = HappyBackgroundGeneration(happy_gen)
    output = happy_bg_gen.generate_background_info("library", context="I like to read books")
    assert type(output.text) == str

def test_bg_gen_simple_with_spacy():
    nlp = spacy.load("en_core_web_sm")
    happy_gen = HappyGeneration("GPT2", "gpt2")
    happy_bg_gen = HappyBackgroundGeneration(happy_gen, nlp)
    output_1 = happy_bg_gen.generate_background_info("library", context="I like to read books")
    output_2 = happy_bg_gen.generate_background_info("library", context="I like to read books")
    print("Singular: ", output_1)
    print("Plural: ", output_2)

def test_bg_gen_custom_settings():
    happy_gen = HappyGeneration("GPT2", "gpt2")
    happy_bg_gen = HappyBackgroundGeneration(happy_gen)
    output = happy_bg_gen.generate_background_info("library", settings=gen_beam_settings, context="I like to read books")
    print(output)
