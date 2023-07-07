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

# Legacy imports


from happytransformer.legacy import (
    ARGS_GEN_TRAIN,
    ARGS_GEN_EVAl,
    ARGS_QA_TRAIN,
    ARGS_QA_EVAl,
    ARGS_QA_TEST,
    ARGS_SP_TRAIN,
    ARGS_TC_TRAIN,
    ARGS_TC_EVAL,
    ARGS_TC_TEST,
    ARGS_TOC_TRAIN,
    ARGS_TOC_EVAl,
    ARGS_TOC_TEST,
    ARGS_WP_TRAIN,
    ARGS_WP_EVAl)



from happytransformer.happy_generation import (
    HappyGeneration, GENSettings
)

name = "happytransformer"
