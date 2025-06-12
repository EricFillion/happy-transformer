# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# from happytransformer import HappyTextClassification

# happy_tc = HappyTextClassification("DISTILBERT", "distilbert-base-uncased", num_labels=2)
# result = happy_tc.eval("examples/sample_data.csv.csv")
# print("Accuracy:", result.accuracy)


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from happytransformer import HappyTextClassification

happy_tc = HappyTextClassification("DISTILBERT", "distilbert-base-uncased", num_labels=2)
result = happy_tc.eval("examples/sample_data.csv.csv")

# Print all metrics
for metric_name, metric_value in result.metrics.items():
    print(f"{metric_name.capitalize()}: {metric_value:.4f}")



