# Implementation of AdvAug: Robust Adversarial Augmentation for Neural Machine Translation (https://arxiv.org/pdf/2006.11834.pdf)
import numpy as np
import torch
from scipy.spatial.distance import cosine
from tqdm import tqdm
# pip install fastBPE regex requests sacremoses subword_nmt


def advaug(inputs, translation_loss=0.2, sampling_ratio=0.25):
    # Tokenize input and generate positions to create perturbations
    assert isinstance(inputs, list), "input must be a list"

    # Load en -> de and de -> en models
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                           checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                           tokenizer='moses', bpe='fastbpe')
    en2de.cuda()
    de2en = torch.hub.load('pytorch/fairseq',
                           'transformer.wmt19.de-en.single_model',
                           tokenizer='moses', bpe='fastbpe')
    de2en.cuda()

    outputs = []
    for input in tqdm(inputs):
        words = input.split()
        random_raw_indexes = np.random.uniform(0, len(words) - 1, round(sampling_ratio * len(words)))
        positions = [round(index) for index in random_raw_indexes]
        output_words = []

        # Iterate through each token in input
        for index in range(0, len(words)):
            if index in positions:
                # Calculate vocab list for word
                en_binarized_word = en2de.binarize(en2de.apply_bpe(words[index]))
                de_word = en2de.translate(words[index])
                en_binarized_perturbations = de2en.generate(de2en.binarize(de2en.apply_bpe(de_word)), beam=10,
                                                            sampling=True, sampling_topk=10)
                vocab_with_dupes = []
                for i, en_binarized_perturbation in enumerate(en_binarized_perturbations):
                    en_perturbation = de2en.decode(en_binarized_perturbations[i]['tokens'])
                    vocab_with_dupes.append(en_perturbation)
                vocab = []
                for i, word in enumerate(vocab_with_dupes):
                    if words[index].lower() != word.lower():
                        if len(word.split()) <= 3:
                            if word != '"' and word != '\'':
                                vocab.append(vocab_with_dupes[i].lower())

                # Calculate gradient for word
                gradient = np.gradient(en_binarized_word) * translation_loss

                # Calculate optimal adversarial perturbation for word
                perturbation_scores = {}
                for i, perturbation in enumerate(vocab):
                    binarized_perturbation = en2de.binarize(en2de.apply_bpe(perturbation))

                    # Attempted padding
                    """size_diff = len(binarized_perturbation) - len(en_binarized_word)
                    print(size_diff)
                    if size_diff > 0:
                        padding = torch.zeros(size_diff)
                        en_binarized_word_padded = np.concatenate((en_binarized_word.numpy(), padding.numpy()), axis=None)
                        perturbation_scores[i] = 1 - cosine((torch.sub(binarized_perturbation, torch.from_numpy(en_binarized_word_padded))).tolist(), gradient.tolist())
                    elif size_diff < 0:
                        padding = torch.zeros(-size_diff)
                        binarized_perturbation_padded = np.concatenate((binarized_perturbation.numpy(), padding.numpy()), axis=None)
                        perturbation_scores[i] = 1 - cosine((torch.sub(torch.from_numpy(binarized_perturbation_padded), en_binarized_word)).tolist(), gradient.tolist())
                    else:
                        perturbation_scores[i] = 1 - cosine((torch.sub(binarized_perturbation, en_binarized_word)).tolist(), gradient.tolist())"""

                    if len(binarized_perturbation) == len(en_binarized_word):
                        perturbation_scores[i] = 1 - cosine((torch.sub(binarized_perturbation, en_binarized_word)).tolist(),
                                                            gradient.tolist())
                if len(perturbation_scores) != 0:
                    output_words.append(vocab[max(perturbation_scores, key=perturbation_scores.get)])
                    # output_words.append(vocab[np.argmax(perturbation_scores)])
                else:
                    output_words.append(words[index])

            else:
                # If words were not selected to be replaced, return the original words
                output_words.append(words[index])

        # Construct and return output sentence
        output = ''
        for word in output_words:
            output = output + word + ' '
        output = output[:-1]
        outputs.append(output)
    
    # Freeing cuda space
    del en2de
    del de2en
    torch.cuda.empty_cache()
    
    return outputs

if __name__ == '__main__':
    text = "Consider an imaginary elephant in the room which is of size 100 x 100 and weight 10,000 pounds which does nothing but sit and eat and sleep"
    print(advaug(text))
