from bwic import data_collection_bwic
from happy_transformer.happy_roberta import HappyRoBERTa

def bwic_roberta():
    """
    Using HappyRoBERTa to solve the Billion Word Imputation Challenge
    Writes the result to bwic_output.txt
    The run time is too long for a standard laptop to complete.
    prints the result to a file called bwic_output.txt
    """
    happy_roberta = HappyRoBERTa()
    data = data_collection_bwic.get_data_bwic()
    data.pop(0) # remove the header line
    output = open("bwic_output.txt", 'w')
    output.write("id,\"sentence\"\n")

    test_case = 0
    for case in data:
        test_case += 1
        case_array = case.split(" ")
        top_softmax = 0
        top_word = ""
        top_index = 0

        for i in range(0, len(case_array)):
            mask_list = case_array.copy()
            mask_list.insert(i, "<mask>")
            mask_sentence = " ".join(mask_list)
            predictions = happy_roberta.predict_k_masks(mask_sentence, 20)
            j = 0

            while j < len(predictions):
                temp_word = predictions[j][0]
                temp_softmax = predictions[j][1]
                if temp_word.isalpha():
                    if temp_softmax > top_softmax:
                        top_index = i
                        top_softmax = temp_softmax
                        top_word = temp_word
                    break
                j += 1

        case_array.insert(top_index, top_word)
        final_answer = " ".join(case_array)
        final_answer = str(test_case) + ",\"" + final_answer+"\"\n"
        output.write(final_answer)
        print(final_answer)


bwic_roberta()
