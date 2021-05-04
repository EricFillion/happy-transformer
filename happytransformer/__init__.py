from happytransformer.happy_question_answering import HappyQuestionAnswering
from happytransformer.happy_word_prediction import HappyWordPrediction
from happytransformer.happy_text_classification import HappyTextClassification
from happytransformer.happy_next_sentence import HappyNextSentence
from happytransformer.happy_token_classification import HappyTokenClassification
from happytransformer.happy_generation import HappyGeneration, GENSettings

from happytransformer.gen.default_args import ARGS_GEN_TRAIN, ARGS_GEN_EVAl
from happytransformer.qa.default_args import ARGS_QA_TRAIN, ARGS_QA_EVAl, ARGS_QA_TEST
from happytransformer.tc.default_args import ARGS_TC_TRAIN, ARGS_TC_EVAL, ARGS_TC_TEST
from happytransformer.wp.default_args import ARGS_WP_TRAIN, ARGS_WP_EVAl


from happytransformer.gen.trainer import GENTrainArgs, GENEvalArgs
from happytransformer.qa.trainer import QATestArgs, QAEvalArgs, QATrainArgs
from happytransformer.tc.trainer import TCTrainArgs, TCEvalArgs, TCTestArgs
from happytransformer.wp.trainer import WPTrainArgs, WPEvalArgs

from happytransformer.happy_generation import (
    HappyGeneration, GENSettings
)

name = "happytransformer"
