from happy_transformer.bert_utils import (train, switch_to_new, create_dataset, evaluate)
from transformers import BertForMaskedLM, BertTokenizer


def fine_tune(train_path, test_path, batch_size=1, epochs=1, lr=5e-5, adam_epsilon=1e-8,
              model_name='bert-base-uncased'):
    """

    :param train_path: Path to the training file, expected to be a .txt or similar
    :param test_path: Path to the testing file, expected to be a .txt or similar

    Default parameters for effortless finetuning
    batch size = 1
    Number of epochs  = 1
    Learning rate = 5e-5
    Adam epsilon = 1e-8
    Model = 'bert-base-uncased'

    """
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))  # To make sure embedding size agrees with the tokenizer

    # Start Train
    model.cuda()
    train_dataset = create_dataset(tokenizer, file_path=train_path)
    train(model, tokenizer, train_dataset, batch_size=batch_size, epochs=epochs, lr=lr, adam_epsilon=adam_epsilon)

    # Start Eval
    model, tokenizer = switch_to_new('model')
    model.cuda()
    test_dataset = create_dataset(tokenizer, file_path=test_path)
    return evaluate(model, tokenizer, test_dataset, batch_size=2)
