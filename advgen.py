import numpy as np

def advgen(input, likelihood_func=0, translation_loss=0, sampling_ratio=0.05):
    words = input.split()
    random_raw_indexes = np.random.uniform(0, len(words) - 1, round(sampling_ratio * len(words)))
    positions = [int(index) for index in random_raw_indexes]
    output_words = []
    for index in range(0, len(words)):
        if words[index] in positions:
            # Implement rest of equation
            a = 1
        else:
            output_words.append(words[index])
    output = ''
    for word in output_words:
        output = output + word + ' '
    output = output[:-1]
    return output


print(advgen(input='This is a test sentence with lots and lots of words in it.'))
