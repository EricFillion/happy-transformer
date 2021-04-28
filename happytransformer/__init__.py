from happytransformer.happy_question_answering import HappyQuestionAnswering
from happytransformer.happy_word_prediction import HappyWordPrediction
from happytransformer.happy_text_classification import HappyTextClassification
from happytransformer.happy_next_sentence import HappyNextSentence
from happytransformer.happy_token_classification import HappyTokenClassification
from happytransformer.happy_generation import HappyGeneration

from happytransformer.gen.default_args import ARGS_GEN_TRAIN, ARGS_GEN_EVAl
from happytransformer.qa.default_args import ARGS_QA_TRAIN, ARGS_QA_EVAl
from happytransformer.tc.default_args import ARGS_TC_TRAIN, ARGS_TC_EVAL
from happytransformer.wp.default_args import ARGS_WP_TRAIN, ARGS_WP_EVAl

from happytransformer.happy_generation import (
    HappyGeneration,
    GEN_DEFAULT_SETTINGS,
    GEN_GREEDY_SETTINGS,
    GEN_BEAM_SETTINGS,
    GEN_TOP_K_SAMPLING_SETTINGS,
    GEN_GENERIC_SAMPLING_SETTINGS,
)

name = "happytransformer"
