from util import prepareData, normalizeString
from torch import optim
from models import EncoderRNN, DecoderRNN
from train_utils import trainIters, get_datasets, load_checkpoint
import torch, argparse, sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_input():
    parser = argparse.ArgumentParser(description="Train an NMT Seq2Seq model")
    parser.add_argument('checkpoint', default='None')
    parser.add_argument('lr', default=1e-4, type=float)
    parser.add_argument('num_epochs', default=50, type=int)
    parser.add_argument('batch_size', default=128, type=int)
    parser.add_argument('hidden_size', default=256, type=int)
    parser.add_argument('num_layers', default=1, type=int)
    parser.add_argument('dropout', default=0.1, type=float)
    parser.add_argument('teacher_forcing_ratio', default=1.0, type=float)
    parser.add_argument('reverse_input', type=bool)

    args = parser.parse_args()

    print("Read checkpoint: {}".format(args.checkpoint))
    print("Read learning rate: {}".format(args.lr))
    print("Read number of epochs: {}".format(args.num_epochs))
    print("Read batch size: {}".format(args.batch_size))
    print("Read hidden size: {}".format(args.hidden_size))
    print("Read number of GRU layers: {}".format(args.num_layers))
    print("Read dropout: {}".format(args.dropout))
    print("Read teacher forcing ratio: {}".format(args.teacher_forcing_ratio))
    print("Read reverse input: {}".format(args.reverse_input))

    return args


if __name__ == "__main__":
    from_lang, to_lang = 'eng', 'fra'
    args = read_input()

    print("Using device {}".format(device))

    #To prepare the vocabularies, use all data
    input_lang, output_lang, _ = prepareData(from_lang, to_lang, False)

    # Get datasets
    try:
        train_set, _ = get_datasets(from_lang, to_lang)
    except Exception as e:
        print("Train dataset file not found. You might have to run split_datasets.py")
        sys.exit(-1)

    hidden_size = args.hidden_size
    encoder = EncoderRNN(input_lang.n_words, args.hidden_size, args.num_layers).to(device)
    decoder = DecoderRNN(output_lang.n_words, args.hidden_size, args.num_layers, dropout_p=args.dropout).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    if args.checkpoint != 'None':
        print("Found checkpoint at {}".format(args.checkpoint))
        encoder, decoder, optimizer, epoch, loss = load_checkpoint(args.checkpoint, encoder, decoder, optimizer)
        trainIters(train_set, input_lang, output_lang, encoder, decoder, args.num_epochs, args.teacher_forcing_ratio,
               optimizer, simplify=False, reverse_input=args.reverse_input, learning_rate=args.lr, batch_size=args.batch_size)
    else:
        print("Training from scratch")
        trainIters(train_set, input_lang, output_lang, encoder, decoder, args.num_epochs, args.teacher_forcing_ratio,
               optimizer, simplify=False, reverse_input=args.reverse_input, learning_rate=args.lr, batch_size=args.batch_size)

    # TODO: Use BLEU score
