import string
import re

def tokenize_sentences(tokenizer, text):
    """
    Formats a sentence so that it can be tokenized by a transformer.
    Required because built-in tokenizer methods assume a single sentence.
    :param text: a 1-2 sentence text that contains [MASK]
    :return: A list of tokens as strings
    """

    # Add sep token when we have word followed by punct followed by whitespace
    text_with_sep = re.sub('(\S+[.!?]\s)', f'\\1 {tokenizer.sep_token}', text)

    return [
        tokenizer.cls_token,
        *tokenizer.tokenize(text_with_sep),
        tokenizer.sep_token
    ]