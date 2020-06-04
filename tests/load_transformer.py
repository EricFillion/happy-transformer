# This test is here to see if we can
# minimize logging
from happytransformer import HappyBERT

transformer = HappyBERT()
predictions = transformer.predict_mask("Dogs make me [MASK] to eat",num_results=20)
# when runnning this, logs should be minimal