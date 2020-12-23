from happytransformer import HappyBERT

happy = HappyBERT('madlag/bert-base-uncased-squad-v1-sparse0.25')

PARAGRAPH = (
    'Jesus was crucified on the cross. '
    'The bible contains many stories such as this one. '
    'A variety of authors detail the events. '
    'Britain has the largest army and France has the longest road. '
)

QA_PAIRS = [
    ('Who has the largest army?', 'Britain'),
    ('Who has the longest road?', 'France'),
    ('How did Jesus die?', 'Crucified on the cross')
]

def test_qa_multi():
    for question, expected_answer in QA_PAIRS:
        computed_answers = happy.answers_to_question(question, PARAGRAPH, k=10)
        computed_answer = happy.answer_question(question, PARAGRAPH)
        assert len(computed_answers) == 10
        assert computed_answers[0].text.lower() == computed_answer.lower() == expected_answer.lower()