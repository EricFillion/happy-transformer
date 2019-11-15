def get_data_bwic():
    test_file = open("test_bwic.txt", "r")
    test_file_rows = test_file.readlines()
    result_array = list()

    for row in test_file_rows:
        starting_index = row.find("\"") + 1 # find the location of the start of the sentence4
        row = row[starting_index:-2] # Each sentence ends with a quotation mark which must be removed
        result_array.append(row)
    print(len(result_array))
    return result_array



