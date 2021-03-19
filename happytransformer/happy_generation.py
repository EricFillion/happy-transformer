"""
Contains the HappyGeneration class
"""
from dataclasses import dataclass
from transformers import AutoModelForCausalLM
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.toc.trainer import TOCTrainer
from happytransformer.adaptors import get_adaptor

DEFAULT_SETTINGS = {
    "do_sample": False,
    "early_stopping": False,
    "num_beams": 1,
    "temperature": 0.65,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1,
    "length_penalty": 1,
    "no_repeat_ngram_size": 2,
    'bad_words_ids': None,
}

default_greedy_settings = {
    "do_sample": False,
    "early_stopping": False,
    "repetition_penalty": 1,
    "length_penalty": 1,
    "no_repeat_ngram_size": 2,
    'bad_words_ids': None,
}


default_beam_settings = {
    "num_beams": 5,
    "early_stopping": True,

    "do_sample": False,
    "repetition_penalty": 1,
    "length_penalty": 1,
    "no_repeat_ngram_size": 2,
    'bad_words_ids': None,
}

default_generic_sampling_settings = {
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 0,
    "early_stopping": False,
    "repetition_penalty": 1,
    "length_penalty": 1,
    "no_repeat_ngram_size": 2,
    'bad_words_ids': None,
}


default_top_k_sampling_settings = {
    "do_sample": True,
    "top_k": 50,
    "early_stopping": False,
    "repetition_penalty": 1,
    "length_penalty": 1,
    "no_repeat_ngram_size": 2,
    'bad_words_ids': None,
}


default_p_nucleus_sampling_settings = {
    "do_sample": True,
    "top_k": 0,
    "top_p": 0.92,
    "early_stopping": False,
    "repetition_penalty": 1,
    "length_penalty": 1,
    "no_repeat_ngram_size": 2,
    'bad_words_ids': None,
}


@dataclass
class GenerationResult:
    text: str


class HappyGeneration(HappyTransformer):
    """
    A user facing class for text generation
    """
    def __init__(self, model_type: str = "GPT2", model_name: str = "gpt2"):

        self.adaptor = get_adaptor(model_type)

        model = AutoModelForCausalLM.from_pretrained(model_name)

        super().__init__(model_type, model_name, model)

        self._trainer = TOCTrainer(self.model, model_type, self.tokenizer, self._device, self.logger)

    def __check_default_text_is_val(self, text):

        if not isinstance(text, str):
            raise ValueError("The text input must be a string")
        elif not text:
            raise ValueError("The text input must have at least one character")


    def generate_text(self, text, settings=default_greedy_settings,
                      min_length=20, max_length=60) -> GenerationResult:
        """
        :param text: starting text that the model uses to generate text with.
        :param method: either one of 1/5 preconfigured methods, or "custom" to indicate custom settings
        :param custom_settings: if method == "custom", then custom settings may be provided in the form of
              a dictionary. Refer to the README to see potential parameters to add.
              Parameters that are not added to the dictionary will keep their default value.
        :return: Text that the model generates.
        """

        self.__check_default_text_is_val(text)

        settings = self.get_settings(settings)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        adjusted_min_length = min_length + len(input_ids[0])
        adjusted_max_length = max_length + len(input_ids[0])
        output = self.model.generate(input_ids,
                                     min_length=adjusted_min_length,
                                     max_length=adjusted_max_length,
                                     **settings
                                     )
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        final_result = self.__gt_post_processing(result, text)

        return GenerationResult(text=final_result)



    def __gt_post_processing(self, result, text):
        """
        A method for processing the output of the model. More features will be added later.
        :param result: result the output of the model after being decoded
        :param text:  the original input to generate_text
        :return: returns to text after going through post-processing. Removes starting text
        """

        return result[len(text):]

    def get_settings(self, custom_settings):

        possible_keys = list(DEFAULT_SETTINGS.keys())
        settings = DEFAULT_SETTINGS.copy()
        for key, value in custom_settings.items():
            if key not in possible_keys:
                self.logger.warning("\"%s\" is not a valid argument", key)
            else:
                settings[key] = custom_settings[key]
        return settings

    def train(self, input_filepath, args):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath):
        raise NotImplementedError("test() is currently not available")
