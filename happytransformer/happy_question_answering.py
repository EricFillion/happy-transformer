"""
Contains the HappyQuestionAnswering class.

"""
from collections import namedtuple
import torch
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.qa.trainer import QATrainer
from happytransformer.qa.default_args \
    import ARGS_QA_TRAIN
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    DistilBertForQuestionAnswering,
    DistilBertTokenizerFast,
    AlbertForQuestionAnswering,
    AlbertTokenizerFast,
    RobertaForQuestionAnswering,
    RobertaTokenizerFast,
    QuestionAnsweringPipeline,
)

QuestionAnsweringResult = namedtuple("QuestionAnsweringResult", ["answer", "score", "start", "end"])


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
                 model_name="distilbert-base-cased-distilled-squad"):
        model = None
        tokenizer = None

        if model_type == "ALBERT":
            model = AlbertForQuestionAnswering.from_pretrained(model_name)
            tokenizer = AlbertTokenizerFast.from_pretrained(model_name)
        elif model_type == "BERT":
            model = BertForQuestionAnswering.from_pretrained(model_name)
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
        elif model_type == "DISTILBERT":
            model = DistilBertForQuestionAnswering.from_pretrained(model_name)
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        elif model_type == "ROBERTA":
            model = RobertaForQuestionAnswering.from_pretrained(model_name)
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

        else:
            raise ValueError(self.model_type_error)

        super().__init__(model_type, model_name, model, tokenizer)
        device_number = 1 if torch.cuda.is_available() else -1
        # from documentation " a positive will run the model on the associated CUDA device id."
        # todo: get device ID if torch.cuda.is_available()

        self._pipeline = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=device_number)

        self._trainer = QATrainer(model, model_type, tokenizer, self._device, self.logger)

    def answer_question(self, context, question, topk=1):
        """
        :param context: background information to answer the question (string)
        :param question: A question that can be answered with the given context (string)
        :param topk: how many results
        :return: A list of a named tuples that contains the keys: answer, score, start and end

        """

        result = self._pipeline(context=context, question=question, topk=topk)
        # transformers returns a single dictionary when topk ==1.
        # Our convention however is to have constant output format
        if topk == 1:
            result = [result]

        results = [
            QuestionAnsweringResult(
                answer=answer["answer"],
                score=answer["score"],
                start=answer["start"],
                end=answer["end"],)
            for answer in result
        ]
        return results

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
        self._trainer.train(input_filepath=input_filepath, args=args)

    def eval(self, input_filepath):
        """
        Trains the question answering model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
        context, question, answer_text, answer_start

        return: A dictionary that contains a key called "eval_loss"

        """
        return self._trainer.eval(input_filepath=input_filepath)

    def test(self, input_filepath):
        """
        Tests the question answering model. Used to obtain results

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
        context, question

        return: A list of dictionaries. Each dictionary
        contains the keys: "score", "start", "end" and "answer"
        """
        return self._trainer.test(input_filepath=input_filepath, solve=self.answer_question)
