from happytransformer.happy_question_answering import HappyQuestionAnswering
from happytransformer.happy_word_prediction import HappyWordPrediction
from happytransformer.happy_text_classification import HappyTextClassification
from happytransformer.happy_next_sentence import HappyNextSentence
from happytransformer.happy_token_classification import HappyTokenClassification
from happytransformer.happy_generation import HappyGeneration, GENSettings
from happytransformer.happy_text_to_text import HappyTextToText, TTSettings

from happytransformer.args import (
    GENTrainArgs, GENEvalArgs,
    QATestArgs, QAEvalArgs, QATrainArgs,
    TCTrainArgs, TCEvalArgs, TCTestArgs,
    WPTrainArgs, WPEvalArgs,
    TTTrainArgs, TTEvalArgs
)

name = "happytransformer"
