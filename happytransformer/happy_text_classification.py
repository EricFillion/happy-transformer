"""
Contains a class called HappyTextClassification that performs text classification
"""
from dataclasses import dataclass

from transformers import TextClassificationPipeline, AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding

from happytransformer.args import TCTrainArgs, TCEvalArgs, TCTestArgs
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.adaptors import get_adaptor
from happytransformer.fine_tuning_util import EvalResult

from tqdm import tqdm
import csv


@dataclass
class TextClassificationResult:
    label: str
    score: float

class HappyTextClassification(HappyTransformer):
    """
    A user facing class for Text Classification
    """

    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-uncased", num_labels: int = 2, load_path: str = "", use_auth_token: str = None, from_tf=False):
        self.adaptor = get_adaptor(model_type)

        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)

        if load_path != "":
            model = AutoModelForSequenceClassification.from_pretrained(load_path, config=config, from_tf=from_tf)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, use_auth_token=use_auth_token, from_tf=from_tf)


        super().__init__(model_type, model_name, model, use_auth_token=use_auth_token, load_path=load_path)

        self._pipeline = TextClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer,
            device=self.device
        )

        self._data_collator = DataCollatorWithPadding(self.tokenizer)
        self._t_data_file_type = "csv"
        self._type = "tc"


    def classify_text(self, text: str) -> TextClassificationResult:
        """
        Classify text to a label based on model's training
        """
        # Blocking allowing a for a list of strings
        if not isinstance(text, str):
            raise ValueError("the \"text\" argument must be a single string")
        results = self._pipeline(text)
        # we do not support predicting a list of  texts, so only first prediction is relevant
        first_result = results[0]

        return TextClassificationResult(label=first_result["label"], score=first_result["score"])

    def train(self, input_filepath: str, eval_filepath: str = "", args: TCTrainArgs =TCTrainArgs()):
        """
        Trains the question answering model
        input_filepath: a string that contains the location of a csv file for training and evaluating if eval_filepath is not provided.  Contains the following header values: text, label
        eval_filepath: a string that contains the location of a csv file for evaluation. Contains the following header values: text, label.
        args: Either a TCTrainArgs() object or a dictionary that contains all of the same keys as ARGS_TC_TRAIN
        return: None
        """
        super(HappyTextClassification, self).train(input_filepath, args, eval_filepath)


    def eval(self, input_filepath, args: TCEvalArgs =TCEvalArgs()) -> EvalResult:
        """
        Evaluated the text classification answering model
        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
        text, label

        return: an EvalResult() object
        """
        return super(HappyTextClassification, self).eval(input_filepath, args)


    def test(self, input_filepath, args=TCTestArgs()):
        """
        Tests the text classification  model. Used to obtain results
        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header value:
         text
        return: A list of TextClassificationResult() objects
        """

        if type(args) == dict:
            raise ValueError("#todo")

        contexts = self._get_data(input_filepath, test_data=True)

        return [
            self.classify_text(context)
            for context in tqdm(contexts)
        ]

    def _tok_function(self, raw_dataset, dataclass_args: TCTrainArgs):

        def __preprocess_function(case):
            result = self.tokenizer(case["text"], truncation=True, padding=True)
            result["labels"] = case["label"]
            return result

        tok_dataset = raw_dataset.map(
            __preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=["text"],
            desc="Tokenizing data"
        )

        return tok_dataset

    def _get_data(self, filepath, test_data=False):
        """
        Used for parsing data for training and evaluating (both contain labels)
        :param filepath: a string that contains the location of the data
        :return:
        """
        contexts = []
        labels = []
        with open(filepath, newline='', encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                contexts.append(row['text'])
                if not test_data:
                    labels.append(int(row['label']))
        csv_file.close()

        if not test_data:
            return contexts, labels
        return contexts
