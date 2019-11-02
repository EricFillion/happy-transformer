import torch
import string
import re
from transformers import XLNetModel, XLNetTokenizer


class Happy_XLNET():
    """

    """

    def __init__(self):

        self.transformer = XLNetModel.from_pretrained("models")
        self.tokenizer = XLNetTokenizer.from_pretrained('tokenizers')
        self.transformer.eval()

        # GPU support
        self.gpu_support = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using model:", self.gpu_support)

    def rank(self, text: str):
        '''
        Predicts the mask.
        '''

        formatted_text = self.__get_formatted_text(text)  # Not working yet
        tokenized_text = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        masked_index = self.__get_prediction_index(tokenized_text)
        segment_id = self.__get_segment_ids(tokenized_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segment_id])

        tokens_tensor = tokens_tensor.to(self.gpu_support)
        segments_tensors = segments_tensors.to(self.gpu_support)

        print(formatted_text)

        with torch.no_grad():
            outputs = self.transformer(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]
            print(predictions)

            softmax = self.softmax(predictions)

            top_prediction = torch.topk(softmax[0, masked_index], 1)
            prediction_softmax = top_prediction[0].tolist()
            prediction_index = top_prediction[1].tolist()

            prediction_token = self.tokenizer.convert_ids_to_tokens(prediction_index)

            if self.gpu_support == "cuda":
                torch.cuda.empty_cache()

            return prediction_token, prediction_softmax

    def __get_prediction_index(self, tokenized_text):
        """
        Gets the location of the first occurrence of the [MASK] index
        :param tokenized_text: a list of word tokens where one of the tokens is the string "[MASK]"
        :return:
        """
        # TODO: put in HappyBERT. Overwrite HappyTransformer.
        #  Maybe only the masked token needs to be changed per HappyClass

        # TODO: easy: there might be a cleaner way to do this

        return tokenized_text.index('<mask>')

        return None  # TODO: medium: find a proper way to deal with errors

    def __get_formatted_text(self, text):
        # TODO: put in HappyBERT. Overwrite HappyTransformer
        """
        Formats a sentence so that BERT it can be tokenized by BERT.
        :param text: a 1-2 sentence text that contains [MASK]
        :return: A string with the same sentence that contains the required tokens for BERT
        """

        # Create a spacing around each punctuation character. eg "!" -> " ! "
        # TODO: easy: find a cleaner way to do punctuation spacing
        text = re.sub('([.,!?()])', r' \1 ', text)
        # text = re.sub('\s{2,}', ' ', text)

        split_text = text.split()
        new_text = list()
        new_text.append("[CLS]")

        for i, char in enumerate(split_text):
            new_text.append(char)
            if char not in string.punctuation:
                pass
            # must be a punctuation symbol
            elif i + 1 >= len(split_text):
                # is the last punctuation so simply add to the new_text
                pass
            else:
                if split_text[i + 1] in string.punctuation:
                    pass
                else:
                    new_text.append("[SEP]")
                # must be a middle punctuation

        new_text.append("[SEP]")
        text = " ".join(new_text)

        return text

    def __get_segment_ids(self, tokenized_text: list):
        # TODO: put in HappyBERT
        """
        Converts a list of tokens into segment_ids. The segment id is a array
        representation of the location for each character in the
        first and second sentence. This method only words with 1-2 sentences.
        Example:
        tokenized_text = ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]',
                         'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        returns segments_ids
        """

        #         split_location = tokenized_text.index('[SEP]')
        #         segment_ids = list()
        #         for i in range(0, len(tokenized_text)):
        #             if i <= split_location:
        #                 segment_ids.append(0)
        #             else:
        #                 segment_ids.append(1)
        #         return segment_ids
        return [0] * len(tokenized_text)  # for a single sentence

    def softmax(self, x):
        return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)


def main():
    happy = Happy_XLNET()
    print(happy.rank("My dog is very <mask>"))


if __name__ == '__main__':
    main()
