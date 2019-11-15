def get_data_bwic():
    """
    :return: a list where each index is a line without "test_bwic.txt" without junk characters
    """
    test_file = open("test_bwic.txt", "r")
    test_file_rows = test_file.readlines()
    result_list = list()

    for row in test_file_rows:
        starting_index = row.find("\"") + 1 # find the location of the start of the sentence4
        row = row[starting_index:-2] # Each sentence ends with a quotation mark which must be removed
        result_list.append(row)
    print(len(result_list))
    return result_list
