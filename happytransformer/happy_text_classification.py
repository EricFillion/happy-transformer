import csv
from typing import Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataclasses import dataclass
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TextClassificationPipeline

from happytransformer.adaptors import get_adaptor
from happytransformer.args import TCEvalArgs, TCTestArgs, TCTrainArgs
from happytransformer.fine_tuning_util import EvalResult
from happytransformer.happy_transformer import HappyTransformer

@dataclass
class TextClassificationResult:
    label: str
    score: float

class HappyTextClassification(HappyTransformer):
    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-uncased", num_labels: int = 2, load_path: str = "", use_auth_token: Union[bool, str] = None,  trust_remote_code: bool=False):

        self._num_labels = num_labels
        self.adaptor = get_adaptor(model_type)
        model_class = AutoModelForSequenceClassification

        super().__init__(model_type, model_name, model_class, use_auth_token=use_auth_token, load_path=load_path, trust_remote_code=trust_remote_code)

        self._pipeline_class = TextClassificationPipeline

        self._data_collator = DataCollatorWithPadding(self.tokenizer)
        self._t_data_file_type = ["csv"]
        self._type = "tc"


    def classify_text(self, text: str) -> TextClassificationResult:

        # loads pipeline if it hasn't been loaded already.
        self._load_pipeline()

        # Blocking allowing a for a list of strings
        if not isinstance(text, str):
            raise ValueError("the \"text\" argument must be a single string")
        results = self._pipeline(text)
        # we do not support predicting a list of  texts, so only first prediction is relevant
        first_result = results[0]

        return TextClassificationResult(label=first_result["label"], score=first_result["score"])

    def train(self, input_filepath: str, args: TCTrainArgs =TCTrainArgs(), eval_filepath: str = ""):
        super(HappyTextClassification, self).train(input_filepath, args, eval_filepath)


    # def eval(self, input_filepath, args: TCEvalArgs =TCEvalArgs()) -> EvalResult:
    #     return super(HappyTextClassification, self).eval(input_filepath, args)
    # def eval(self, input_filepath, args: TCEvalArgs = TCEvalArgs()) -> EvalResult:
    #     contexts, labels = self._get_data(input_filepath)

    #     predictions = []
    #     for text in tqdm(contexts, desc="Evaluating"):
    #         result = self.classify_text(text)
    #         # result.label is like "LABEL_0", "LABEL_1", so extract the number
    #         label_index = int(result.label.split("_")[-1])
    #         predictions.append(label_index)

    #     correct = sum([int(p == l) for p, l in zip(predictions, labels)])
    #     accuracy = correct / len(labels)

        
    #     return EvalResult(loss=None, accuracy=accuracy)
    def eval(self, input_filepath, args: TCEvalArgs = TCEvalArgs()) -> EvalResult:
        contexts, labels = self._get_data(input_filepath)

        predictions = []
        for text in tqdm(contexts, desc="Evaluating"):
            result = self.classify_text(text)
            label_index = int(result.label.split("_")[-1])
            predictions.append(label_index)

        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        return EvalResult(loss=None, metrics=metrics)






    def test(self, input_filepath, args=TCTestArgs()):
        self.logger.warning(f"HappyTextClassification.test() is deprecated and will be removed in version 4.0.0.")

        if type(args) == dict:
            raise ValueError("#todo")

        contexts = self._get_data(input_filepath, test_data=True)

        return [
            self.classify_text(context)
            for context in tqdm(contexts)
        ]

    def _tok_function(self, raw_dataset, args: TCTrainArgs, file_type: str) -> Dataset:

        def __preprocess_function(case):
            result = self.tokenizer(case["text"], truncation=True, padding=True)
            result["labels"] = case["label"]
            return result

        tok_dataset = raw_dataset.map(
            __preprocess_function,
            batched=True,
            remove_columns=["text", "label"],
            desc="Tokenizing data"
        )

        return tok_dataset

    def _get_data(self, filepath, test_data=False):
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

    def _get_model_components(self, model_name_path,  use_auth_token, trust_remote_code, model_class):
        # HappyTextClassification is the only class that overwrites
        # this as we need to specify number of labels.
        config = AutoConfig.from_pretrained(model_name_path, token=use_auth_token,  num_labels=self._num_labels)
        model = model_class.from_pretrained(model_name_path, config=config, token=use_auth_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name_path, token=use_auth_token)

        return config, tokenizer, model