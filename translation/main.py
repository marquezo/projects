from util import prepareData
from models import EncoderRNN, DecoderRNN
from train_utils import trainIters
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    input_lang, output_lang, pairs = prepareData('eng', 'fra', False)
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder1 = DecoderRNN(output_lang.n_words, hidden_size, dropout_p=0.1).to(device)

    trainIters(pairs, input_lang, output_lang, encoder1, decoder1, 100000, print_every=5000, batch_size=128)

    torch.save(encoder1, 'encoder_no_att.model')
    torch.save(decoder1, 'decoder_no_att.model')

    #TODO: split data into train, dev, test; change code to handle minibatches, and use BLEU score

if __name__ == "__main__": main()