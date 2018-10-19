from util import prepareData
from models import EncoderRNN, DecoderRNN
from train_utils import load_checkpoint, evaluateRandomly, get_datasets
import torch, argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_input():
    parser = argparse.ArgumentParser(description="Evaluate an NMT Seq2Seq model")
    parser.add_argument('checkpoint')
    parser.add_argument('num_sentences', type=int)

    args = parser.parse_args()

    print("Read checkpoint: {}".format(args.checkpoint))
    print("Read number of sentences: {}".format(args.num_sentences))

    return args


if __name__ == "__main__":
    from_lang, to_lang = 'eng', 'fra'
    args = read_input()

    print("Using device {}".format(device))
    # To prepare the vocabularies, use all data
    input_lang, output_lang, pairs = prepareData(from_lang, to_lang, False)

    # Get datasets
    _, dev_set = get_datasets(from_lang, to_lang)

    hidden_size = 256
    encoder = EncoderRNN(input_lang.n_words, hidden_size, 1).to(device)
    decoder = DecoderRNN(output_lang.n_words, hidden_size, 1, dropout_p=0.1).to(device)

    encoder, decoder, _, _, _ = load_checkpoint(args.checkpoint, encoder, decoder, None)

    evaluateRandomly(dev_set, input_lang, output_lang, encoder, decoder, False, args.num_sentences)

    #TODO: Use BLEU score
