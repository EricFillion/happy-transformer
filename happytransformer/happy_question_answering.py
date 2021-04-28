"""
Contains the HappyQuestionAnswering class.

"""
from typing import List
from dataclasses import dataclass
from transformers import QuestionAnsweringPipeline, AutoModelForQuestionAnswering

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.qa.trainer import QATrainer, QATrainArgs, QAEvalArgs, QATestArgs
from happytransformer.happy_trainer import EvalResult
from happytransformer.qa import ARGS_QA_TRAIN, ARGS_QA_EVAl, ARGS_QA_TEST

from happytransformer.cuda_detect import detect_cuda_device_number
from happytransformer.adaptors import get_adaptor
from happytransformer.fine_tuning_util import create_args_dataclass

@dataclass
class QuestionAnsweringResult:
    answer: str
    score: float
    start: int
    end: int


class HappyQuestionAnswering(HappyTransformer):
    """
    This class is a user facing class that allows users to solve question answering problems using
    a transformer QA models. These models are able to answer a question given context for the
    question by selecting a span within the context that answers the question.

    The purpose of this class is to be lightweight and easy
    to understand and to offload complex tasks to
    other classes.
    """
    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-cased-distilled-squad", load_path: str = ""):
        
        self.adaptor = get_adaptor(model_type)

        if load_path != "":
            model = AutoModelForQuestionAnswering.from_pretrained(load_path)
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        super().__init__(model_type, model_name, model)
        device_number = detect_cuda_device_number()

        self._pipeline = QuestionAnsweringPipeline(model=self.model, tokenizer=self.tokenizer, device=device_number)

        self._trainer = QATrainer(self.model, model_type, self.tokenizer, self._device, self.logger)

    def answer_question(self, context: str, question: str, top_k: int = 1) \
            -> List[QuestionAnsweringResult]:
        """
        Find the answers to a question.
        The answer MUST be contained somewhere within the context for this to work.
        top_k describes the number of answers to return.
        """

        pipeline_output = self._pipeline(context=context, question=question, topk=top_k)
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

    def train(self, input_filepath, args=ARGS_QA_TRAIN):
        """
        Trains the question answering model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values: context,
        question, answer_text, answer_start

        args: a dictionary that contains settings found under
        happytransformer.happytasks.happy_qa.default_args.py

        return: None
        """

        method_dataclass_args = create_args_dataclass(default_dic_args=ARGS_QA_TRAIN,
                                                            input_dic_args=args,
                                                            method_dataclass_args=QATrainArgs)

        self._trainer.train(input_filepath=input_filepath, dataclass_args=method_dataclass_args)

    def eval(self, input_filepath, args=ARGS_QA_EVAl) -> EvalResult:
        """
        Trains the question answering model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
        context, question, answer_text, answer_start

        return: A dictionary that contains a key called "eval_loss"

        """
        method_dataclass_args = create_args_dataclass(default_dic_args=ARGS_QA_EVAl,
                                                            input_dic_args=args,
                                                            method_dataclass_args=QAEvalArgs)
        return self._trainer.eval(input_filepath=input_filepath, dataclass_args=method_dataclass_args)


    def test(self, input_filepath, args=ARGS_QA_TEST):
        """
        Tests the question answering model. Used to obtain results

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
        context, question

        return: A list of dictionaries. Each dictionary
        contains the keys: "score", "start", "end" and "answer"
        """
        method_dataclass_args =  create_args_dataclass(default_dic_args=ARGS_QA_TEST,
                                                            input_dic_args=args,
                                                            method_dataclass_args=QATestArgs)
        return self._trainer.test(input_filepath=input_filepath, solve=self.answer_question, dataclass_args=method_dataclass_args)
