from happytransformer import HappyBERT

happy = HappyBERT('bert-large-uncased-whole-word-masking-finetuned-squad')

PARAGRAPH = (
    'McGill is a university located in Montreal. '
    'It was founded in 1821, making it the eight oldest university in Canada. '
    'It is currently ranked 31st worldwide according to the QS Global World Ranking '

)

QA_PAIRS = [
    ('When was McGill founded?', '1821'),
    ('Where is McGill located?', 'Montreal'),
    ('What is McGill\'s worldwide ranking?', '31st'),

]

def test_qa_multi():
    for question, expected_answer in QA_PAIRS:
        computed_answers = happy.answers_to_question(question, PARAGRAPH, k=10)
        computed_answer = happy.answer_question(question, PARAGRAPH)
        # k is being respected
        assert len(computed_answers) == 10
        # both answering methods yield correct result
        assert computed_answers[0].text.lower() == expected_answer.lower()
        assert computed_answer.lower() == expected_answer.lower()
        
        total_p = sum(answer.probability for answer in computed_answers)
        # probabilties for answers_to_question() add up to 1 ish
        assert abs(total_p-1) < 0.01