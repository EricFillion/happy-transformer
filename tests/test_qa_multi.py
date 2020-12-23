from happytransformer import HappyBERT

happy = HappyBERT('madlag/bert-base-uncased-squad-v1-sparse0.25')

PARAGRAPH = (
    'Jesus was crucified on the cross. '
    'The bible contains many stories such as this one. '
    'A variety of authors detail the events. '
    'Britian has the largest army and France has the longest road. '
)
QUESTION = 'How did Jesus die?'

def test_qa_multi():
    answers = happy.answers_to_question(QUESTION, PARAGRAPH)
    print(answers)
    print()
    answer = happy.answer_question(QUESTION, PARAGRAPH)
    print(answer)