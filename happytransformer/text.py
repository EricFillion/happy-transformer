def remove_starting_character(self, options, starting_char):
    new_predictions = list()
    for prediction in options:
        if prediction[0] == starting_char:
            new_prediction = prediction[1:]
            new_predictions.append(new_prediction)
        else:
            new_predictions.append(prediction)
    return new_predictions