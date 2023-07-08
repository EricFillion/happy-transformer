from dataclasses import dataclass
from transformers import Text2TextGenerationPipeline, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.adaptors import get_adaptor
from happytransformer.fine_tuning_util import EvalResult
from happytransformer.args import TTTrainArgs, TTEvalArgs, TTTestArgs
from typing import Union

@dataclass
class TextToTextResult:
    text: str

@dataclass
class TTSettings:
    min_length: int = 10
    max_length: int = 50
    do_sample: bool = False
    early_stopping: bool = False
    num_beams: int = 1
    temperature: float = 1
    top_k: int = 50
    no_repeat_ngram_size: int = 0
    top_p: float = 1


class HappyTextToText(HappyTransformer):
    """
    A user facing class for text to text generation
    """
    def __init__(self, model_type: str = "T5", model_name: str = "t5-small", load_path: str = "", use_auth_token: str = None, from_tf=False):

        self.adaptor = get_adaptor(model_type)
        model_class = AutoModelForSeq2SeqLM

        super().__init__(model_type, model_name, model_class, use_auth_token=use_auth_token, load_path=load_path)


        self._pipeline = Text2TextGenerationPipeline(model=self.model,
                                                     tokenizer=self.tokenizer, device=self.device)

        self.__max_input_length = 1024
        self.__max_output_length = 1024

        self._data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        self._t_data_file_type = "csv"
        self._type = "tt"


    def __assert_default_text_is_val(self, text):
        if not isinstance(text, str):
            raise ValueError("The text input must be a string")
        if not text:
            raise ValueError("The text input must have at least one character")


    def generate_text(self, text: str,
                      args: TTSettings = TTSettings()) -> TextToTextResult:
        self.__assert_default_text_is_val(text)

        output = self._pipeline(text, min_length=args.min_length,
                                max_length=args.max_length,
                                do_sample=args.do_sample,
                                early_stopping=args.early_stopping,
                                num_beams=args.num_beams,
                                temperature=args.temperature,
                                top_k=args.top_k,
                                no_repeat_ngram_size=args.no_repeat_ngram_size,
                                top_p=args.top_p,
                                )
        return TextToTextResult(text=output[0]['generated_text'])

    def train(self, input_filepath, eval_filepath: str = "", args: TTTrainArgs=TTTrainArgs()):
        super(HappyTextToText, self).train(input_filepath, args, eval_filepath)

    def eval(self, input_filepath, args=TTEvalArgs()) -> EvalResult:
        return super(HappyTextToText, self).eval(input_filepath, args)

    def test(self, input_filepath, args=TTTestArgs):
        raise NotImplementedError("test() is currently not available")


    def _tok_function(self, raw_dataset, dataclass_args: Union[TTTrainArgs, TTEvalArgs]):

        self.__max_input_length = dataclass_args.max_input_length
        self.__max_output_length = dataclass_args.max_output_length

        def __preprocess_function(examples):
            model_inputs = self.tokenizer(examples["input"], max_length=self.__max_input_length, truncation=True)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(examples["target"], max_length=self.__max_output_length, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tok_dataset = raw_dataset.map(
            __preprocess_function,
            batched=True,
            num_proc=dataclass_args.preprocessing_processes,
            remove_columns=["input", "target"],
        )

        return tok_dataset