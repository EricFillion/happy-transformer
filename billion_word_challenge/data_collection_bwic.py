import os
def get_data_bwic():
    """
    :return: a list where each index is a line without "test_bwic.txt" without junk characters

    """
    os.chdir("billion_word_challenge")
    test_file = open("test_bwic.txt", "r")
    os.chdir("..")
    test_file_rows = test_file.readlines()
    result_list = list()

    for row in test_file_rows:
        starting_index = row.find("\"") + 1 # find the location of the start of the sentence4
        row = row[starting_index:-2] # Each sentence ends with a quotation mark which must be removed
        result_list.append(row)
    print("Number of Billion Word Challenge test cases:", len(result_list) - 1)
    result_list.pop(0) # Removes the header
    return result_list
