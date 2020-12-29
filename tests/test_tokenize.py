from happytransformer import HappyBERT
from happytransformer.tokenize import tokenize_sentences

happy_bert = HappyBERT('bert-base-uncased')

def test_tokenize_two_sentences():
    text = 'Hello, traveller. Welcome to the hotel.'
    computed_tokens = tokenize_sentences(happy_bert.tokenizer, text)
    expected_tokens = [
        '[CLS]',
        'hello',',','traveller','.',
        '[SEP]',
        'welcome','to','the','hotel','.',
        '[SEP]'
    ]
    assert computed_tokens == expected_tokens