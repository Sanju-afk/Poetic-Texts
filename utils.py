import numpy as np
import random

# Sampling function to add randomness to predictions
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Text generation function
def generate_text(model, text, length, temperature, SEQ_LENGTH, char_to_index, index_to_char):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ""
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    for i in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(char_to_index)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_pred, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated
