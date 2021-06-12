"""
Contains the HappyGeneration class
"""
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, TextGenerationPipeline
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.gen.trainer import GENTrainer, GENTrainArgs, GENEvalArgs
from happytransformer.adaptors import get_adaptor
from happytransformer.gen import ARGS_GEN_TRAIN, ARGS_GEN_EVAl, ARGS_GEN_TEST
from happytransformer.happy_trainer import EvalResult
from happytransformer.fine_tuning_util import create_args_dataclass
from happytransformer.cuda_detect import detect_cuda_device_number

"""
The main settings that users will adjust when performing experiments

The values for full_settings are the same as the default values above except for min and max length. 
"""
@dataclass
class GENSettings:
    min_length: int = 10
    max_length: int = 50
    do_sample: bool = False
    early_stopping: bool = False
    num_beams: int = 1
    temperature: float = 1
    top_k: int = 50
    no_repeat_ngram_size: int = 0
    top_p: float = 1

@dataclass
class GenerationResult:
    text: str


class HappyGeneration(HappyTransformer):
    """
    This class is a user facing class that allows users to generate text using
    text generation Transformer models.

    The purpose of this class is to be lightweight and easy
    to understand and to offload complex tasks to
    other classes.
    """
    def __init__(self, model_type: str = "GPT2", model_name: str = "gpt2", load_path: str = ""):

        self.adaptor = get_adaptor(model_type)

        if load_path != "":
            model = AutoModelForCausalLM.from_pretrained(load_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)

        super().__init__(model_type, model_name, model)
        device_number = detect_cuda_device_number()

        self._pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer, device=device_number)

        self._trainer = GENTrainer(self.model, model_type, self.tokenizer, self._device, self.logger)

    def __assert_default_text_is_val(self, text):
        """
        Ensures the input's text input is valid.
        Raises a Value Error if the text input is not valid.
        :param text: The value the user inputs for the "text" parameter
        """

        if not isinstance(text, str):
            raise ValueError("The text input must be a string")
        if not text:
            raise ValueError("The text input must have at least one character")


    def generate_text(self, text: str, args: GENSettings=GENSettings()) -> GenerationResult:
        """
        :param text: starting text that the model uses as a prompt to continue it.
        :param args: A GENSettings object
        :return: A GenerationResult() object
        """

        self.__assert_default_text_is_val(text)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        adjusted_min_length = args.min_length + len(input_ids[0])
        adjusted_max_length = args.max_length + len(input_ids[0])

        output = self._pipeline(text, min_length=adjusted_min_length,
                                return_full_text=False,
                                max_length=adjusted_max_length,
                                do_sample=args.do_sample,
                                early_stopping=args.early_stopping,
                                num_beams=args.num_beams,
                                temperature=args.temperature,
                                top_k=args.top_k,
                                no_repeat_ngram_size=args.no_repeat_ngram_size,
                                top_p=args.top_p
                                )
        return GenerationResult(text=output[0]['generated_text'])


    def __post_process_generated_text(self, result, text):
        """
        A method for processing the output of the model. More features will be added later.
        :param result: result the output of the model after being decoded
        :param text:  the original input to generate_text
        :return: returns to text after going through post-processing. Removes starting text
        """

        return result[len(text):]


    def train(self, input_filepath: str, args=GENTrainArgs()):
        """
        :param input_filepath:a file path to a text file that contains nothing but training data
        :param args: either a GENTrainArgs() object or a dictionary that contains all of the same keys as ARGS_GEN_TRAIN
        :return: None
        """

        if type(args) == dict:
            method_dataclass_args = create_args_dataclass(default_dic_args=ARGS_GEN_TRAIN,
                                                         input_dic_args=args,
                                                         method_dataclass_args=GENTrainArgs)
        elif type(args) == GENTrainArgs:
            method_dataclass_args = args
        else:
            raise ValueError("Invalid args type. Use a GENTrainArgs object or a dictionary")

        self._trainer.train(input_filepath=input_filepath, dataclass_args=method_dataclass_args)

    def eval(self, input_filepath: str, args=GENEvalArgs()) -> EvalResult:
        """
        :param input_filepath:a file path to a text file that contains nothing but evaluating data
        :param args: either a GENEvalArgs() object or a dictionary that contains all of the same keys as ARGS_GEN_EVAl
        :return: None
        """
        if type(args) == dict:
            method_dataclass_args = create_args_dataclass(default_dic_args=ARGS_GEN_EVAl,
                                                         input_dic_args=args,
                                                         method_dataclass_args=GENEvalArgs)
        elif type(args) == GENEvalArgs:
            method_dataclass_args = args
        else:
            raise ValueError("Invalid args type. Use a GENEvalArgs object or a dictionary")

        return self._trainer.eval(input_filepath=input_filepath, dataclass_args=method_dataclass_args)

    def test(self, input_filepath, args=ARGS_GEN_TEST):
        raise NotImplementedError("test() is currently not available")
