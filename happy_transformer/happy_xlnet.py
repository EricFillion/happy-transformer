# pylint: disable=W0511
from transformers import XLNetLMHeadModel, XLNetTokenizer

from happy_transformer.happy_transformer import HappyTransformer
from happy_transformer.sequence_classification import SequenceClassifier
from happy_transformer.classifier_utils import classifier_args
import logging


import os

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)




from happy_transformer.classifier_utils import convert_examples_to_features, processors


class HappyXLNET(HappyTransformer):
    """
    Implementation of XLNET for masked word prediction
    """

    def __init__(self, model='xlnet-large-cased', initial_transformers=[]):
        super().__init__(model, initial_transformers)
        self.mlm = None
        self.seq = None
        self.model_version = model
        self.tokenizer = XLNetTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.model = 'XLNET'
        self.classifier_name = ""



        self.seq_args = classifier_args.copy()
        self.seq_trained = False


    def _get_masked_language_model(self):
        """
        Initializes the XLNetLMHeadModel transformer
        """
        self.mlm = XLNetLMHeadModel.from_pretrained(self.model_to_use)
        self.mlm.eval()


    def _init_sequence_classifier(self, classifier_name: str):
        self.classifier_name = classifier_name
        self.seq_args['classifier_name'] = classifier_name
        self.seq_args['model_name'] = classifier_name
        self.seq_args['model_name'] = self.model_version
        self.seq_args['output_dir'] = "outputs/" + classifier_name
        self.seq_args['cache_dir'] = "cache/" + classifier_name
        self.seq_args['data_dir'] = "data/" + classifier_name
        self.seq = SequenceClassifier(self.seq_args)
        print(self.classifier_name, "has been initialized")


    def _train_sequence_classifier(self, train_df):
        if self.seq == None:
            print("First initialize the sequence classifier")
            return
        #  self.seq.train_tsv = train_df.to_csv(sep='\t', index=True, header=False, columns=train_df.columns)
        train_df = train_df.astype("str")
        self.seq.train_list_data = train_df.values.tolist()


        print(type(self.seq.train_list_data))
        print(self.seq.train_list_data )
        self.seq.train_dataset = self.load_and_cache_examples(task="binary", tokenizer=self.tokenizer, evaluate=False)
        self.seq_args["do_train"] = True
        self.seq.run_sequence_classifier()
        self.seq_args["do_train"] = False
        self.seq_trained = True

        print("Training for ", self.classifier_name, "has been completed")


    def _eval_sequence_classifier(self, eval_df):
        if self.seq_trained == False:
            print("First train the sequence classifier")
            return
        eval_df = eval_df.astype("str")


        self.seq.eval_list_data = eval_df.values.tolist()


        self.seq.eval_dataset = self.load_and_cache_examples(task="binary", tokenizer=self.tokenizer, evaluate=True)


        self.seq_args["do_eval"] = True
        self.seq.run_sequence_classifier()
        self.seq_args["do_eval"] = False
        print("Evaluation for ", self.classifier_name, "has been completed")



    def load_and_cache_examples(self, task, tokenizer, evaluate):

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        self.seq.processor = processors[task]()
        output_mode = self.seq_args['output_mode']

        if not self.seq.features_exists and not self.seq_args['reprocess_input_data']:
            logger.info("Loading features from cached file %s")


        else:
            self.features_exists = True
            logger.info("Creating features from dataset file at %s", self.seq_args['cache_dir'])
            label_list = self.seq.processor.get_labels()

            if evaluate:
                examples = self.seq.processor.get_dev_examples(self.seq.eval_list_data)
            else:

                examples = self.seq.processor.get_train_examples(self.seq.train_list_data)


            print("examples: ", examples)
            print(type(examples))

            # if __name__ == "__main__":
            self.seq.features = convert_examples_to_features(examples, label_list, self.seq_args['max_seq_length'], tokenizer,
                                                    output_mode,
                                                    cls_token_at_end=bool(self.seq_args['model_type'] in ['xlnet']),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if self.seq_args['model_type'] in [
                                                        'xlnet'] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(self.seq_args['model_type'] in ['roberta']),
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    pad_on_left=bool(self.seq_args['model_type'] in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if self.seq_args['model_type'] in [
                                                        'xlnet'] else 0)

        all_input_ids = torch.tensor([f.input_ids for f in self.seq.features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.seq.features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.seq.features], dtype=torch.long)
        all_label_ids = None
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in self.seq.features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in self.seq.features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset
