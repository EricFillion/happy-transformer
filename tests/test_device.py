from tests import happy_gen
from happytransformer import HappyGeneration

def test_pipeline_init():
    happy = HappyGeneration("GPT-2", "gpt2")
    assert happy._pipeline is None

    assert not happy._on_device

    happy.generate_text("Hello world ")

    assert happy._pipeline is not None
    assert happy._on_device
