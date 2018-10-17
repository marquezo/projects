from util import prepareData, normalizeString
from models import EncoderRNN, DecoderRNN
from train_utils import trainIters
import torch
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_datasets(lang1, lang2):
    train_file = "data/%s-%s.train" % (lang1, lang2)
    dev_file = "data/%s-%s.dev" % (lang1, lang2)

    if os.path.isfile(train_file) and os.path.isfile(dev_file):

        print("Loading existing datasets found in {} and {}".format(train_file, dev_file))

        # Read the file and split into lines
        train_lines = open(train_file, encoding='utf-8').read().strip().split('\n')
        dev_lines = open(dev_file, encoding='utf-8').read().strip().split('\n')

        # Split every line into pairs and normalize
        train_pairs = [[normalizeString(s) for s in l.split('\t')] for l in train_lines]
        dev_pairs = [[normalizeString(s) for s in l.split('\t')] for l in dev_lines]

        return train_pairs, dev_pairs
    else:
        raise Exception("Files not found")


def split_save_datasets(pairs, lang1, lang2, split_percentage=0.1):
    valid_size = int(len(pairs) * split_percentage)
    valid_set_indices = np.random.choice(len(pairs), valid_size, replace=False)
    print("Size of train/dev set to create is {}/{}".format(len(pairs) - len(valid_set_indices), len(valid_set_indices)))
    valid_set_indices.sort()  # sort so that we pad efficiently

    dev_pairs= []
    train_pairs = []

    for idx in range(len(pairs)):
        if idx in valid_set_indices:
            dev_pairs.append(pairs[idx])
        else:
            train_pairs.append(pairs[idx])

    del pairs

    print("Size of created train/dev set is {}/{}".format(len(train_pairs), len(dev_pairs)))

    train_file = "data/%s-%s.train" % (lang1, lang2)
    dev_file = "data/%s-%s.dev" % (lang1, lang2)

    with open(train_file, 'w') as f:
        for pair in train_pairs:
            f.write("{}\t{}\n".format(pair[0], pair[1]))

    with open(dev_file, 'w') as f:
        for pair in dev_pairs:
            f.write("{}\t{}\n".format(pair[0], pair[1]))

    print("Wrote datasets to {} and {}".format(train_file, dev_file))

    return train_pairs, dev_pairs


def main():

    from_lang, to_lang = 'eng', 'fra'

    print("Using device {}".format(device))
    # To prepare the vocabularies, use all data
    input_lang, output_lang, pairs = prepareData(from_lang, to_lang, False)

    # Get datasets
    try:
        train_set, _ = get_datasets(from_lang, to_lang)
    except Exception as e:
        train_set, _ = split_save_datasets(pairs, from_lang, to_lang)

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder1 = DecoderRNN(output_lang.n_words, hidden_size, dropout_p=0.1).to(device)

    trainIters(train_set, input_lang, output_lang, encoder1, decoder1, 20, 1.0, reverse_input=True, batch_size=128)

    #TODO: Use BLEU score

if __name__ == "__main__": main()
