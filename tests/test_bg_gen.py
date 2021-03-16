from happytransformer import HappyBackgroundGeneration, HappyGeneration
import spacy

def test_bg_gen_simple():
    happy_gen = HappyGeneration("GPT2", "gpt2")
    happy_bg_gen = HappyBackgroundGeneration(happy_gen)
    output_1 = happy_bg_gen.generate_background_info("library", context="I like to read books", method="greedy")
    print("Singular: ", output_1)

def test_bg_gen_simple_with_spacy():
    nlp = spacy.load("en_core_web_sm")
    happy_gen = HappyGeneration("GPT2", "gpt2")
    happy_bg_gen = HappyBackgroundGeneration(happy_gen, nlp)
    output_1 = happy_bg_gen.generate_background_info("library", context="I like to read books", method="greedy")
    output_2 = happy_bg_gen.generate_background_info("library", context="I like to read books", method="greedy")
    print("Singular: ", output_1)
    print("Plural: ", output_2)
