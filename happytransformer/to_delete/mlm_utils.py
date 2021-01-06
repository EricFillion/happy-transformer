"""
BERT and ROBERTA masked language model fine-tuning:

Credit: This code is a modified version of the code found in this repository
under
        https://github.com/huggingface/transformers/blob/master/examples
        /run_lm_finetuning.py

"""

import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from tqdm import trange
from tqdm.notebook import tqdm_notebook
from transformers import (AdamW)

try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    from transformers import WarmupLinearSchedule \
        as get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """
    Used to turn .txt file into a suitable dataset object
    """

    def __init__(self, tokenizer, file_path, block_size=512):
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        lines = text.split("\n")
        self.examples = []
        for line in lines:
            tokenized_text = tokenizer.encode(line, max_length=block_size,
                                              add_special_tokens=True, pad_to_max_length=True)  # Get ids from text
            self.examples.append(tokenized_text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def set_seed(seed=42):
    """
    Sets seed for all random number generators available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        print('Cuda manual seed is not set')


def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling:
    80% MASK, 10% random, 10% original.
    * The standard implementation from Huggingface Transformers library *
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    # MLM Prob is 0.15 in examples
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in
        labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(
        special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with
    # tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens
    # unchanged
    return inputs, labels


def train(model, tokenizer, train_dataset, batch_size, lr, adam_epsilon,
          epochs):
    """

    :param model: Bert Model to train
    :param tokenizer: Bert Tokenizer to train
    :param train_dataset:
    :param batch_size: Stick to 1 if not using using a high end GPU
    :param lr: Suggested learning rate from paper is 5e-5
    :param adam_epsilon: Used for weight decay fixed suggested parameter is
    1e-8
    :param epochs: Usually a single pass through the entire dataset is
    satisfactory
    :return: Loss
    """

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size)

    t_total = len(train_dataloader) // batch_size  # Total Steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 0, t_total)

    # ToDo Case for fp16

    # Start of training loop
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", batch_size)

    model.train()
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    train_iterator = trange(int(epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            inputs, labels = mask_tokens(batch, tokenizer)
            inputs = inputs.to('cuda')  # Don't bother if you don't have a gpu
            labels = labels.to('cuda')

            outputs = model(inputs, masked_lm_labels=labels)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            loss.backward()
            tr_loss += loss.item()

            # if (step + 1) % 1 == 0: # 1 here is a placeholder for gradient
            # accumulation steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    return model, tokenizer


def create_dataset(tokenizer, file_path, block_size=512):
    """
    Creates a dataset object from file path.
    :param tokenizer: Bert tokenizer to create dataset
    :param file_path: Path where data is stored
    :param block_size: Should be in range of [0,512], viable choices are 64,
    128, 256, 512
    :return: The dataset
    """
    dataset = TextDataset(tokenizer, file_path=file_path,
                          block_size=block_size)
    return dataset


def evaluate(model, tokenizer, eval_dataset, batch_size):
    """

    :param model: Newly trained Bert model
    :param tokenizer:Newly trained Bert tokenizer
    :param eval_dataset:
    :param batch_size: More flexible than training, the user can get away
    with picking a higher batch_size
    :return: The perplexity of the dataset
    """
    eval_sampler = SequentialSampler(eval_dataset)  # Same order samplinng
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    # Evaluation loop

    for batch in tqdm_notebook(eval_dataloader, desc='Evaluating'):
        inputs, labels = mask_tokens(batch, tokenizer)
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss)).item()

    result = {
        'perplexity': perplexity,
        'eval_loss': eval_loss
    }

    return result


word_prediction_args = {
    "batch_size": 1,
    "epochs": 1,
    "lr": 5e-5,
    "adam_epsilon": 1e-8

}


class FinetuneMlm():
    """

    :param train_path: Path to the training file, expected to be a .txt or
    similar
    :param test_path: Path to the testing file, expected to be a .txt or
    similar

    Default parameters for effortless finetuning
    batch size = 1
    Number of epochs  = 1
    Learning rate = 5e-5
    Adam epsilon = 1e-8
    model_name = Must be in default transformer name. ie: 'bert-base-uncased'

    """

    def __init__(self, mlm, args, tokenizer, logger):
        self.mlm = mlm
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger

    def train(self, train_path):
        self.mlm.resize_token_embeddings(len(self.tokenizer))
        # Start Train
        self.mlm.cuda()
        train_dataset = create_dataset(
            self.tokenizer, file_path=train_path)
        self.mlm, self.tokenizer = train(self.mlm, self.tokenizer,
                                         train_dataset,
                                         batch_size=self.args["batch_size"],
                                         epochs=self.args["epochs"],
                                         lr=self.args["lr"],
                                         adam_epsilon=self.args[
                                             "adam_epsilon"])

        del train_dataset
        self.mlm.cpu()
        return self.mlm, self.tokenizer

    def evaluate(self, test_path, batch_size):
        self.mlm.cuda()
        test_dataset = create_dataset(self.tokenizer, file_path=test_path)
        result = evaluate(self.mlm, self.tokenizer, test_dataset,
                          batch_size=batch_size)
        del test_dataset
        self.mlm.cpu()
        return result
