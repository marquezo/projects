from util import prepareData, normalizeString
from models import EncoderRNN, DecoderRNN
from train_utils import trainIters, get_datasets, split_save_datasets
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    #TODO parse args
    #TODO build a model with more capacity

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

    trainIters(train_set, input_lang, output_lang, encoder1, decoder1, 20, 1.0, True, reverse_input=True, batch_size=128)

    #TODO: Use BLEU score

if __name__ == "__main__": main()
