def softmax(value):
    return value.exp() / (value.exp().sum(-1)).unsqueeze(-1)