import csv
from dataclasses import dataclass
from typing import List, Union

from datasets import Dataset
from transformers import AutoModelForQuestionAnswering, DataCollatorWithPadding, QuestionAnsweringPipeline
from tqdm import tqdm

from happytransformer.adaptors import get_adaptor
from happytransformer.args import QAEvalArgs, QATestArgs, QATrainArgs
from happytransformer.fine_tuning_util import EvalResult
from happytransformer.happy_transformer import HappyTransformer

@dataclass
class QuestionAnsweringResult:
    answer: str
    score: float
    start: int
    end: int


class HappyQuestionAnswering(HappyTransformer):
    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-cased-distilled-squad", 
                 load_path: str = "",
                 use_auth_token: Union[bool, str] = None,
                 trust_remote_code: bool=False
                 ):
        
        self.adaptor = get_adaptor(model_type)
        model_class = AutoModelForQuestionAnswering

        super().__init__(model_type, model_name, model_class,  use_auth_token=use_auth_token, load_path=load_path, trust_remote_code=trust_remote_code)

        self._pipeline_class = QuestionAnsweringPipeline

        self._data_collator = DataCollatorWithPadding(self.tokenizer)

        self._t_data_file_type = ["csv"]

        self._type = "qa"

    def answer_question(self, context: str, question: str, top_k: int = 1) \
            -> List[QuestionAnsweringResult]:

        # loads pipeline if it hasn't been loaded already.
        self._load_pipeline()

        pipeline_output = self._pipeline(context=context, question=question, top_k=top_k)
        # transformers returns a single dictionary when top_k ==1.
        # Our convention however is to have constant output format
        answers = [pipeline_output] if top_k == 1 else pipeline_output

        return [
            QuestionAnsweringResult(
                answer=answer["answer"],
                score=answer["score"],
                start=answer["start"],
                end=answer["end"],)
            for answer in answers
        ]

    def train(self, input_filepath, args: QATrainArgs =QATrainArgs(), eval_filepath: str = ""):
        super(HappyQuestionAnswering, self).train(input_filepath, args, eval_filepath)

    def eval(self, input_filepath, args=QAEvalArgs()) -> EvalResult:

        return super(HappyQuestionAnswering, self).eval(input_filepath, args)


    def test(self, input_filepath, args=QATestArgs()):
        if type(args) == dict:
            raise ValueError( "As of version 2.5.0 dictionary inputs are not acceptable. Please provide a QATestArgs. ")

        contexts, questions = self._get_data(input_filepath, test_data=True)

        return [
            self.answer_question(context, question)[0]
            for context, question in
            tqdm(zip(contexts, questions))
        ]

    def _tok_function(self, raw_dataset, args: QATrainArgs, file_type: str) -> Dataset:

        def __preprocess_function(case):
            case["answer_start"] = int(case['answer_start'])
            gold_text = case['answer_text']
            start_idx = case['answer_start']
            end_idx = start_idx + len(gold_text)

            # todo (maybe): strip answer['text'] (remove white space from start and end)
            # sometimes squad answers are off by a character or two – fix this
            if case["context"][start_idx:end_idx] == gold_text:
                case['answer_end'] = end_idx
            elif case["context"][start_idx - 1:end_idx - 1] == gold_text:
                case["context"]['answer_start'] = start_idx - 1
                case["context"]['answer_end'] = end_idx - 1
            elif case[start_idx - 2:end_idx - 2] == gold_text:
                case["context"]['answer_start'] = start_idx - 2
                case["context"]['answer_end'] = end_idx - 2

            encodings = self.tokenizer(case["context"], case["question"], truncation=True, padding=True)

            start_position = encodings.char_to_token(case['answer_start'])
            end_position =  encodings.char_to_token(case['answer_end'] - 1)
            if start_position is None:
                start_position = self.tokenizer.model_max_length
            if end_position is None:
                end_position = self.tokenizer.model_max_length

            encodings.update({'start_positions': start_position, 'end_positions': end_position})

            return encodings


        tok_dataset = raw_dataset.map(
            __preprocess_function,
            batched=False,
            remove_columns=["context", "question", "answer_text", "answer_start"],
            desc="Tokenizing data"
        )

        return tok_dataset

    @staticmethod
    def _get_data(filepath, test_data=False):
        contexts = []
        questions = []
        answers = []
        with open(filepath, newline='', encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                contexts.append(row['context'])
                questions.append(row['question'])
                if not test_data:
                    answer = {}
                    answer["answer_text"] = row['answer_text']
                    answer["answer_start"] = int(row['answer_start'])
                    answers.append(answer)
        csv_file.close()

        if not test_data:
            return contexts, questions, answers
        return contexts, questions
