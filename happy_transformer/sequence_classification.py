from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import random
import json
import math
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm_notebook, trange

from tensorboardX import SummaryWriter


from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr

from happy_transformer.classifier_utils import convert_examples_to_features, output_modes, processors
import ipywidgets



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceClassifier():



    def __init__(self, args):
        self.args = args
        self.processor = None
        self.device = None
        self.run_sequence_classifier()


    def run_sequence_classifier(self):

        MODEL_CLASSES = {
            'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
            'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
            'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
            'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
        }


        with open('args.json', 'w') as f:
            json.dump(self.args, f)

        if os.path.exists(self.args['output_dir']) and os.listdir(self.args['output_dir']) and self.args['do_train'] and not self.args[
            'overwrite_output_dir']:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    self.args['output_dir']))


        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args['model_type']]

        config = config_class.from_pretrained(self.args['model_name'], num_labels=2, finetuning_task=self.args['task_name'])
        tokenizer = tokenizer_class.from_pretrained(self.args['model_name'])

        model = model_class.from_pretrained(self.args['model_name'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device);

        task = self.args['task_name']

        if task in processors.keys() and task in output_modes.keys():
            processor = processors[task]()
            label_list = processor.get_labels()
            num_labels = len(label_list)
        else:
            raise KeyError(f'{task} not found in processors or in output_modes. Please check utils.py.')

        if self.args['do_train']:
            train_dataset = self.load_and_cache_examples(task, tokenizer)
            global_step, tr_loss = self.train(train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if self.args['do_train']:
            if not os.path.exists(self.args['output_dir']):
                os.makedirs(self.args['output_dir'])
            logger.info("Saving model checkpoint to %s", self.args['output_dir'])

            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(self.args['output_dir'])
            tokenizer.save_pretrained(self.args['output_dir'])
            torch.save(self.args, os.path.join(self.args['output_dir'], 'training_self.args.bin'))

        results = {}
        if self.args['do_eval']:
            checkpoints = [self.args['output_dir']]
            if self.args['eval_all_checkpoints']:
                checkpoints = list(os.path.dirname(c) for c in
                                   sorted(glob.glob(self.args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                model = model_class.from_pretrained(checkpoint)
                model.to(device)
                result, wrong_preds = self.evaluate(model, tokenizer, prefix=global_step)
                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                results.update(result)

        print(results)


    def load_and_cache_examples(self, task, tokenizer, evaluate=False):
        global processor
        processor = processors[task]()
        output_mode = self.args['output_mode']

        mode = 'dev' if evaluate else 'train'
        cached_features_file = os.path.join(self.args['data_dir'],
                                            f"cached_{mode}_{self.args['model_name']}_{self.args['max_seq_length']}_{task}")

        if os.path.exists(cached_features_file) and not self.args['reprocess_input_data']:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)

        else:
            logger.info("Creating features from dataset file at %s", self.args['data_dir'])
            label_list = processor.get_labels()
            examples = processor.get_dev_examples(self.args['data_dir']) if evaluate else processor.get_train_examples(
                self.args['data_dir'])

            #if __name__ == "__main__":
            features = convert_examples_to_features(examples, label_list, self.args['max_seq_length'], tokenizer, output_mode,
                                                    cls_token_at_end=bool(self.args['model_type'] in ['xlnet']),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if self.args['model_type'] in ['xlnet'] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(self.args['model_type'] in ['roberta']),
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    pad_on_left=bool(self.args['model_type'] in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if self.args['model_type'] in ['xlnet'] else 0)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset


    def train(self, train_dataset, model, tokenizer):
        global processor

        tb_writer = SummaryWriter()

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args['train_batch_size'])

        t_total = len(train_dataloader) // self.args['gradient_accumulation_steps'] * self.args['num_train_epochs']

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = math.ceil(t_total * self.args['warmup_ratio'])
        self.args['warmup_steps'] = warmup_steps if self.args['warmup_steps'] == 0 else self.args['warmup_steps']

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args['learning_rate'], eps=self.args['adam_epsilon'])
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.args['warmup_steps'], t_total=t_total)

        if self.args['fp16']:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args['fp16_opt_level'])

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", self.args['num_train_epochs'])
        logger.info("  Total train batch size  = %d", self.args['train_batch_size'])
        logger.info("  Gradient Accumulation steps = %d", self.args['gradient_accumulation_steps'])
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(self.args['num_train_epochs']), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.args['model_type'] in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                print("\r%f" % loss, end='')

                if self.args['gradient_accumulation_steps'] > 1:
                    loss = loss / self.args['gradient_accumulation_steps']

                if self.args['fp16']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args['max_grad_norm'])

                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args['max_grad_norm'])

                tr_loss += loss.item()
                if (step + 1) % self.args['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if self.args['logging_steps'] > 0 and global_step % self.args['logging_steps'] == 0:
                        # Log metrics
                        if self.args[
                            'evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                            results, _ = self.evaluate(model, tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / self.args['logging_steps'], global_step)
                        logging_loss = tr_loss

                    if self.args['save_steps'] > 0 and global_step % self.args['save_steps'] == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(self.args['output_dir'], 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

        return global_step, tr_loss / global_step


    def get_mismatched(self, labels, preds):
        global processor
        mismatched = labels != preds
        examples = processor.get_dev_examples(self.args['data_dir'])
        wrong = [i for (i, v) in zip(examples, mismatched) if v]

        return wrong


    def get_eval_report(self, labels, preds):
        mcc = matthews_corrcoef(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        return {
                   "mcc": mcc,
                   "tp": tp,
                   "tn": tn,
                   "fp": fp,
                   "fn": fn
               }, self.get_mismatched(labels, preds)


    def compute_metrics(self, task_name, preds, labels):
        assert len(preds) == len(labels)
        return self.get_eval_report(labels, preds)


    def evaluate(self, model, tokenizer, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = self.args['output_dir']

        results = {}
        EVAL_TASK = self.args['task_name']

        eval_dataset = self.load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args['eval_batch_size'])

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", self.args['eval_batch_size'])
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(self. device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.args['model_type'] in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if self.args['output_mode'] == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.args['output_mode'] == "regression":
            preds = np.squeeze(preds)
        result, wrong = self.compute_metrics(EVAL_TASK, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        return results, wrong