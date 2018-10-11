from util import prepareData
from models import EncoderRNN, DecoderRNN
from train_utils import trainIters
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    print("Using device {}".format(device))
    input_lang, output_lang, pairs = prepareData('eng', 'fra', False)

    # 10% for validation set
    valid_size = int(len(pairs) * 0.1)
    valid_set_indices = np.random.choice(len(pairs), valid_size, replace=False)
    print("Size of training/validation set is {}/{}".format(len(pairs) - len(valid_set_indices), len(valid_set_indices)))
    valid_set_indices.sort() # sort so that we pad efficiently
    
    valid_set = []
    train_set = []

    for idx in range(len(pairs)):
        if idx in valid_set_indices:
            valid_set.append(pairs[idx])
        else:
            train_set.append(pairs[idx])

    del pairs

    print("Size of training/validation set is {}/{}".format(len(train_set), len(valid_set)))

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder1 = DecoderRNN(output_lang.n_words, hidden_size, dropout_p=0.1).to(device) 

    trainIters(train_set, input_lang, output_lang, encoder1, decoder1, 10, batch_size=128)

    torch.save(encoder1, 'encoder_no_att.model')
    torch.save(decoder1, 'decoder_no_att.model')

    #TODO: Use BLEU score

if __name__ == "__main__": main()
