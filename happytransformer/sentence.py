import re

def is_one_sentence(text):
    """
    Used to verify the proper input requirements for sentence_relation.
    The text must contain no more than a single sentence.
    Casual use of punctuation is accepted, such as using multiple exclamation marks.
    :param text: A body of text
    :return: True if the body of text contains a single sentence, else False
    """
    split_text = re.split('[?.!]', text)
    sentence_found = False
    for possible_sentence in split_text:
        for char in possible_sentence:
            if char.isalpha():
                if sentence_found:
                    return False
                sentence_found = True
                break
    return True