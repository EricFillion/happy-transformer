from happytransformer import HappyGeneration, HappyTextToText, HappyWordPrediction,  GENTrainArgs, TTTrainArgs, WPTrainArgs


def test_pipeline_init():
    happy_gen = HappyGeneration("GPT-2", "sshleifer/tiny-gpt2")
    assert happy_gen._pipeline is None

    assert not happy_gen._on_device

    happy_gen.generate_text("Hello world ")

    assert happy_gen._pipeline is not None
    assert happy_gen._on_device
