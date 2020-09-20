import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def advgen(input, translation_loss=0, sampling_ratio=0.05, d_model=512):
    # Tokenize input and generate positions to create perturbations for
    words = input.split()
    random_raw_indexes = np.random.uniform(0, len(words) - 1, round(sampling_ratio * len(words)))
    positions = [round(index) for index in random_raw_indexes]
    output_words = []

    # Load en -> de and de -> en models
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                           checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                           tokenizer='moses', bpe='fastbpe')
    en2de.cuda()
    de2en = torch.hub.load('pytorch/fairseq',
                           'transformer.wmt19.de-en.single_model',
                           tokenizer='moses', bpe='fastbpe')
    de2en.cuda()

    # Iterate through each token in input
    for index in range(0, len(words)):
        if index in positions:
            # Calculate vocab list for word
            en_bpe = en2de.apply_bpe(words[index])
            en_bin = en2de.binarize(en_bpe)
            de_bin = en2de.generate(en_bin, beam=5, sampling=True, sampling_topk=10)
            de_tokens = de_bin['tokens']
            vocab = []
            for perturbation in de_tokens:
                de_bpe = en2de.string(perturbation)
                de_tokens = en2de.remove_bpe(de_bpe)
                perturbation = en2de.detokenize(de_tokens) # en2de.decode(de_word)
                vocab.append(perturbation)
            if words[index] in vocab:
                vocab.remove(words[index])

            # Calculate gradient for word
            gradient = np.gradient(en_bin) * translation_loss

            # Calculate optimal adversarial perturbation for word
            all_perturbations = []
            for perturbation in vocab:
                perturbation_encoding = en2de.binarize(en2de.apply_bpe(perturbation))
                all_perturbations.append(cosine_similarity((perturbation_encoding - en_bin), gradient))
            output_words.append(vocab[np.argmax(all_perturbations)])
        else:
            # If words were not selected to be replaced, return the original words
            output_words.append(words[index])

    # Construct and return output sentence
    output = ''
    for word in output_words:
        output = output + word + ' '
    output = output[:-1]
    return output


print(advgen(input='This is a test sentence with lots and lots of words in it.'))