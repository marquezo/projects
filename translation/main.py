from util import prepareData, normalizeString
from torch import optim
from models import EncoderRNN, DecoderRNN
from train_utils import trainIters, get_datasets, load_checkpoint
import torch, argparse, sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_input():
    parser = argparse.ArgumentParser(description="Train an NMT Seq2Seq model")
    parser.add_argument('name', type=str)
    parser.add_argument('--checkpoint', default='None')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--teacher_forcing_ratio', default=1.0, type=float)
    parser.add_argument('--reverse_input', action='store_true')
    parser.add_argument('--attention', dest='use_attention', action='store_true')
    parser.add_argument('--simplify', dest='simplify', action='store_true')
    parser.add_argument('--save_loc', default='models')
    parser.set_defaults(feature=True)

    args = parser.parse_args()

    print("Read experiment name: {}".format(args.name))
    print("Read checkpoint: {}".format(args.checkpoint))
    print("Read learning rate: {}".format(args.lr))
    print("Read number of epochs: {}".format(args.num_epochs))
    print("Read batch size: {}".format(args.batch_size))
    print("Read hidden size: {}".format(args.hidden_size))
    print("Read number of GRU layers: {}".format(args.num_layers))
    print("Read dropout: {}".format(args.dropout))
    print("Read teacher_forcing_ratio: {}".format(args.teacher_forcing_ratio))
    print("Read reverse_input: {}".format(args.reverse_input))
    print("Read use_attention: {}".format(args.use_attention))
    print("Read simplify: {}".format(args.simplify))
    print("Read save_loc: {}".format(args.save_loc))

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
    encoder = EncoderRNN(input_lang.n_words, args.hidden_size, args.num_layers, dropout_p=args.dropout).to(device)
    decoder = DecoderRNN(output_lang.n_words, args.hidden_size, args.num_layers, dropout_p=args.dropout,
                         use_attention=args.use_attention).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    if args.checkpoint != 'None':
        print("Found checkpoint at {}".format(args.checkpoint))
        encoder, decoder, optimizer, epoch, loss = load_checkpoint(args.checkpoint, encoder, decoder, optimizer)
        trainIters(args.name, train_set, input_lang, output_lang, encoder, decoder, args.num_epochs, args.teacher_forcing_ratio,
               optimizer, simplify=args.simplify, reverse_input=args.reverse_input, use_attention=args.use_attention,
                   learning_rate=args.lr, batch_size=args.batch_size, save_loc=args.save_loc)
    else:
        print("Training from scratch")
        trainIters(args.name, train_set, input_lang, output_lang, encoder, decoder, args.num_epochs, args.teacher_forcing_ratio,
               optimizer, simplify=args.simplify, reverse_input=args.reverse_input, use_attention=args.use_attention,
                   learning_rate=args.lr, batch_size=args.batch_size, save_loc=args.save_loc)

    # TODO: Use BLEU score
