from happytransformer import HappyBERT

happy = HappyBERT()

PARAGRAPH = (
    'Jesus was crucified on the cross. '
    'The bible contains many stories such as this one. '
    'A variety of authors detail the events. '
)

def test_qa_multi():
    answers = happy.answers_to_question('How did Jesus die?', PARAGRAPH)
    print(answers)