"""
Contains the HappyQuestionAnswering class.

"""

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
    QuestionAnsweringPipeline,
)



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
        if model_type == "BERT":
            model = BertForQuestionAnswering.from_pretrained(model_name)
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
        elif model_type == "DISTILBERT":
            model = DistilBertForQuestionAnswering.from_pretrained(model_name)
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        elif model_type == "ALBERT":
            model = AlbertForQuestionAnswering.from_pretrained(model_name)
            tokenizer = AlbertTokenizerFast.from_pretrained(model_name)
        else:
            raise ValueError("model_type must be BERT or DISTILBERT")

        super().__init__(model_type, model_name, model, tokenizer)
        device_number = 1 if torch.cuda.is_available() else -1
        # from documentation " a positive will run the model on the associated CUDA device id."
        # todo: get device ID if torch.cuda.is_available()

        self._pipeline = QuestionAnsweringPipeline(model, tokenizer, device=device_number)


        self._trainer = QATrainer(model, model_type, tokenizer, self._device,
                                  self.logger)

    def answer_question(self, context, question, topk=1):
        """

        :param context: background information to answer the question (string)
        :param question: A question that can be answered with the given context (string)
        :param topk: how many results
        :return: if topk =1, a dictionary that contains the keys: score, start, end and answer
        if topk >1, a list of dictionaries described above
        """
        return self._pipeline(context=context, question=question, topk=topk)

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
        return self._trainer.eval(input_filepath=input_filepath,)

    def test(self, input_filepath):
        """
        Tests the question answering model. Used to obtain results

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
        context, question

        return: A list of dictionaries. Each dictionary
        contains the keys: "score", "start", "end" and "answer"
        """
        return self._trainer.test(input_filepath=input_filepath, pipeline=self._pipeline)
