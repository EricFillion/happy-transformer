''' This module organizes example WS problems from cs.nyu.edu'''
import xml.etree.ElementTree as et
import pandas as pd


def get_data():
    " Organizes the data into a panda dataframe"
    tree = et.parse('data/WSCollection.xml')

    root = tree.getroot()

    columns = ["txt1", 'pron', 'txt2', 'quote1', 'quote2', 'OptionA', 'OptionB', 'answer']

    dataframe = pd.DataFrame(columns=columns)

    for schema in root:
        problem = dict()
        for element in schema:
            i = 0
            for value in element:

                if value.tag == 'txt1':
                    problem['txt1'] = value.text

                elif value.tag == 'pron':
                    problem['pron'] = value.text

                elif value.tag == 'txt2':
                    problem['txt2'] = value.text

                elif value.tag == 'quote1':
                    problem['quote1'] = value.text

                elif value.tag == 'quote2':
                    problem['quote2'] = value.text

                elif value.tag == 'answer' and i == 0:

                    problem['OptionA'] = value.text
                    i = i + 1

                elif value.tag == 'answer' and i == 1:
                    problem['OptionB'] = value.text

                elif value.tag == "answer":
                    problem['answer'] = value.text

        problem['answer'] = schema.find("correctAnswer").text
        dataframe = dataframe.append(problem, ignore_index=True)
    return dataframe
