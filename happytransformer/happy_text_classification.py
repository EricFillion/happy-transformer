from happytransformer.tc.trainer import TCTrainer

from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,


)

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.tc.default_args import ARGS_TC_EVAL, ARGS_TC_TEST, ARGS_TC_TRAIN

class HappyTextClassification(HappyTransformer):
    def __init__(self, model_type="BERT",
                 model_name="'bert-large-uncased-whole-word-masking-finetuned-squad'", device=None):
        model = None
        tokenizer = None

        if model_type == "BERT":
            model = BertForSequenceClassification.from_pretrained(model_name)
            tokenizer = BertTokenizerFast.from_pretrained(model_name)


        super().__init__(model_type, model_name, model, tokenizer, device)

        self._trainer = TCTrainer(model,
                                  model_type, tokenizer, self._device, self.logger)

    def predict_text(self, text):
        raise NotImplementedError()

    def train(self, input_filepath, args=ARGS_TC_TRAIN):
        """
        Trains the question answering model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values: text,
         label
        args: a dictionary that contains settings found under
        happytransformer.happytasks.happy_qa.default_args.py
        return: None
        """
        self._trainer.train(input_filepath=input_filepath, args=args)

    def eval(self, input_filepath, output_filepath, args=ARGS_TC_EVAL):
        """
        Trains the question answering model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
         text, label
        args: a dictionary that contains settings found under
        happytransformer.happytasks.happy_qa.default_args.py
        output_filepath: a path to a csv file to output the results.
        This file contains the following header values: text,
        label, output, correct, softmax
        return: correct ration (correct/total)
        """
        return self._trainer.eval(input_filepath=input_filepath,
                                  solve=self.predict_text, args=args,
                                  output_filepath=output_filepath)

    def test(self, input_filepath, output_filepath, args=ARGS_TC_TEST):
        """
        Tests the text classification  model. Used to obtain results

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header value:
         text
        args: a dictionary that contains settings found under
        happytransformer.happytasks.happy_qa.default_args.py
        output_filepath: a path to a csv file to output the results.
        This file contains the following header values: text, output, softmax
        return: None
        """
        self._trainer.test(input_filepath=input_filepath,
                           solve=self.predict_text, args=args,
                           output_filepath=output_filepath)
