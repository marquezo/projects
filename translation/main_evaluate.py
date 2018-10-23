from util import prepareData
from models import EncoderRNN, DecoderRNN
from train_utils import load_checkpoint, evaluateRandomly, get_datasets
import torch, argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_input():
    parser = argparse.ArgumentParser(description="Evaluate an NMT Seq2Seq model")
    parser.add_argument('checkpoint')
    parser.add_argument('num_sentences', type=int)
    parser.add_argument('hidden_size', type=int)
    parser.add_argument('num_layers', type=int)
    parser.add_argument('--reverse-input', dest='reverse_input', action='store_true')
    parser.add_argument('--no-reverse-input', dest='reverse_input', action='store_false')
    parser.add_argument('--attention', dest='use_attention', action='store_true')
    parser.add_argument('--no-attention', dest='use_attention', action='store_false')
    parser.add_argument('--simplify', dest='simplify', action='store_true')
    parser.add_argument('--no-simplify', dest='simplify', action='store_false')
    parser.set_defaults(feature=True)

    args = parser.parse_args()

    print("Read checkpoint: {}".format(args.checkpoint))
    print("Read number of sentences: {}".format(args.num_sentences))
    print("Read hidden size: {}".format(args.hidden_size))
    print("Read number of layers: {}".format(args.num_layers))
    print("Read reverse input: {}".format(args.reverse_input))
    print("Read use_attention: {}".format(args.use_attention))
    print("Read simplify: {}".format(args.simplify))

    return args


if __name__ == "__main__":
    from_lang, to_lang = 'eng', 'fra'
    args = read_input()

    print("Using device {}".format(device))
    # To prepare the vocabularies, use all data
    input_lang, output_lang, pairs = prepareData(from_lang, to_lang, False)

    # Get datasets
    _, dev_set = get_datasets(from_lang, to_lang)

    encoder = EncoderRNN(input_lang.n_words, args.hidden_size, args.num_layers).to(device)
    decoder = DecoderRNN(output_lang.n_words, args.hidden_size, args.num_layers, dropout_p=0.1, use_attention=args.use_attention).to(device)

    encoder, decoder, _, _, _ = load_checkpoint(args.checkpoint, encoder, decoder, None)

    evaluateRandomly(dev_set, input_lang, output_lang, encoder, decoder, args.simplify, args.num_sentences, args.reverse_input,
                     args.use_attention)

    #TODO: Use BLEU score
