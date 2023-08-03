from dataclasses import dataclass
from typing import List, Union

from datasets import Dataset
from transformers import AutoModelForCausalLM, default_data_collator, TextGenerationPipeline

from happytransformer.adaptors import get_adaptor
from happytransformer.args import GENEvalArgs, GENTrainArgs
from happytransformer.fine_tuning_util import csv_tok_text_gen_mlm, EvalResult, tok_text_gen_mlm
from happytransformer.happy_transformer import HappyTransformer

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
    bad_words: List[str] = None

@dataclass
class GenerationResult:
    text: str


class HappyGeneration(HappyTransformer):
    def __init__(self, model_type: str = "GPT2", model_name: str = "gpt2", 
                 load_path: str = "", use_auth_token:  Union[bool, str]  = None, trust_remote_code: bool =False):

        self.adaptor = get_adaptor(model_type)
        model_class = AutoModelForCausalLM

        super().__init__(model_type, model_name, model_class,  use_auth_token=use_auth_token, load_path=load_path, trust_remote_code=trust_remote_code)

        self._data_collator = default_data_collator

        self._t_data_file_type = ["text", "csv"]

        self._type = "gen"

        self._pipeline_class = TextGenerationPipeline

    def load_model(self):
        pass


    def __assert_default_text_is_val(self, text):
        if not isinstance(text, str):
            raise ValueError("The text input must be a string")
        if not text:
            raise ValueError("The text input must have at least one character")


    def generate_text(self, text: str, args: GENSettings=GENSettings()) -> GenerationResult:

        # loads pipeline if it hasn't been loaded already.
        self._load_pipeline()

        self.__assert_default_text_is_val(text)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        adjusted_min_length = args.min_length + len(input_ids[0])
        adjusted_max_length = args.max_length + len(input_ids[0])
        if args.bad_words:
            bad_words_ids = [self.tokenizer(" "+phrase.strip()).input_ids for phrase in args.bad_words]
        else:
            bad_words_ids = None

        output = self._pipeline(text, min_length=adjusted_min_length,
                                return_full_text=False,
                                max_length=adjusted_max_length,
                                do_sample=args.do_sample,
                                early_stopping=args.early_stopping,
                                num_beams=args.num_beams,
                                temperature=args.temperature,
                                top_k=args.top_k,
                                no_repeat_ngram_size=args.no_repeat_ngram_size,
                                top_p=args.top_p,
                                bad_words_ids=bad_words_ids
                                )
        return GenerationResult(text=output[0]['generated_text'])


    def __post_process_generated_text(self, result, text):
        return result[len(text):]


    def train(self, input_filepath: str,  args: GENTrainArgs =GENTrainArgs(), eval_filepath: str =""):
        super(HappyGeneration, self).train(input_filepath, args, eval_filepath)

    def eval(self, input_filepath: str, args: GENEvalArgs =GENEvalArgs()) -> EvalResult:
        return super(HappyGeneration, self).eval(input_filepath, args)

    def test(self, input_filepath, args=None):
        raise NotImplementedError("test() is currently not available")

    def _tok_function(self, raw_dataset: Dataset, args: Union[GENTrainArgs, GENEvalArgs], file_type: str) -> Dataset:

        if file_type == "text":
            return tok_text_gen_mlm(tokenizer=self.tokenizer, dataset=raw_dataset,  args=args,
                                           mlm=False)
        else:
            return csv_tok_text_gen_mlm(tokenizer=self.tokenizer, dataset=raw_dataset, args=args,
                                        mlm=False,
                                        )


