from util import prepareData
from models import EncoderRNN, DecoderRNN
from train_utils import load_checkpoint, evaluateRandomly, get_datasets
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    from_lang, to_lang = 'eng', 'fra'

    print("Using device {}".format(device))
    # To prepare the vocabularies, use all data
    input_lang, output_lang, pairs = prepareData(from_lang, to_lang, False)

    # Get datasets
    _, dev_set = get_datasets(from_lang, to_lang)

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder1 = DecoderRNN(output_lang.n_words, hidden_size, dropout_p=0.1).to(device)

    encoder1, decoder1, _, _, _, _ = load_checkpoint("checkpoint.tar", encoder1, decoder1, None, None)

    evaluateRandomly(dev_set, input_lang, output_lang, encoder1, decoder1, True, 1)

    #TODO: Use BLEU score

if __name__ == "__main__": main()
