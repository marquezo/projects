import torch
import random, time, os, math
from util import indexesFromSentence, filterPairs, normalizeString
from lang import SOS_token, EOS_token, PAD_token
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensorFromSentence(lang, sentence, prepend_sos = False):
    if prepend_sos:
        indexes = [SOS_token]
    else:
        indexes = []

    indexes.extend(indexesFromSentence(lang, sentence))
    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype=torch.long, device=device)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1], True)
    return input_tensor, target_tensor


def get_minibatch(pairs, input_lang, output_lang, reverse_input=False):
    """
    :param pairs: list of lists, the inner lists have two elements, the first one contains the input sentence and the
                second one contains the output sentences
    :return: two tensors: one for the source language and the other for the target language.
    """
    list_input_sentences = []
    list_output_sentences = []

    # Go through each pair and keep track of largest sentence to adjust the padding at the end
    for pair in pairs:
        input_tensor = tensorFromSentence(input_lang, pair[0])

        if reverse_input:
            input_tensor = input_tensor.flip(0)

        output_tensor = tensorFromSentence(output_lang, pair[1], True)
        list_input_sentences.append(input_tensor)
        list_output_sentences.append(output_tensor)

    return pad_sequence(list_input_sentences, batch_first=True), pad_sequence(list_output_sentences, batch_first=True)


def train(input_tensor, target_tensor, encoder, decoder, optimizer,
          criterion, teacher_forcing_ratio, use_attention=False):
    """
    :param input_tensor: [batch_size x input_seq_len]
    :param target_tensor: [batch_size x output_seq_len]
    :param encoder:
    :param decoder:
    :param optimizer:
    :param criterion:
    :param teacher_forcing_ratio: For only teacher forcing, pass 1.0
    :return:
    """
    optimizer.zero_grad()
    input_length = input_tensor.size(0)
    encoder_hidden = encoder.initHidden(input_length)
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    decoder_hidden = encoder_hidden

    loss = 0

    if use_attention:
        # encoder_hidden: 1 x batch_size x hidden_size
        # For each hidden state in the minibatch
        # encoder_output: batch_size x seq_len x hidden_size
        target_length = target_tensor.size(1) - 1  # ignore EOS token

        for idx_target in range(target_length):
            targets = target_tensor[:, idx_target]

            # Teacher forcing: Feed the target as the next input, ignore <EOS> token at the end
            decoder_output, decoder_hidden = decoder(targets, decoder_hidden, encoder_output)

            # To calculate loss, ignore <SOS> token, so do +1. It will not overflow because target_length = seq_len - 1
            # Need to do unsqueeze because PyTorch wants it like that
            target_tensor_idx = target_tensor[:, idx_target + 1].unsqueeze(1)
            # Need to swap the dimensions not corresponding to the minibatch for NLLoss to work
            loss_batch = criterion(decoder_output.transpose(1, 2), target_tensor_idx)
            loss += loss_batch.sum()

        # Remember that we padded the target tensor, so find out the number of outputs
        # Sum the loss and divide by the number of outputs
        loss = loss / torch.nonzero(target_tensor[:, 1:]).size(0)
    else:
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input, ignore <EOS> token at the end
            decoder_output, decoder_hidden = decoder(target_tensor[:, :-1], decoder_hidden)

            # Need to swap the dimensions not corresponding to the minibatch for NLLoss to work
            # To calculate loss, ignore <SOS> token
            loss_batch = criterion(decoder_output.transpose(1, 2), target_tensor[:, 1:])

            # Remember that we padded the target tensor, so find out the number of outputs
            # Sum the loss and divide by the number of outputs
            loss = loss_batch.sum() / torch.nonzero(target_tensor[:, 1:]).size(0)
        else:
            # Without teacher forcing: use its own predictions as the next input
            # encoder_hidden: 1 x batch_size x hidden_size
            # For each hidden state in the minibatch
            # For as many tokens in the target until finding EOS, pass the previous output
            # In this case, we don't set decoder_hidden = encoder_hidden because we use decoder_hidden below
            num_values = 0

            for sample_idx in range(encoder_hidden.size(1)): # for each sample in the minibatch
                decoder_hidden = encoder_hidden[0][sample_idx].view(1, 1, -1)
                decoder_input = torch.tensor([[SOS_token]], device=device)

                sample_target_tensor = target_tensor[sample_idx, 1:]

                for di in range(sample_target_tensor.size(0)):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(1) #Get the most probable word
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    decoder_output = decoder_output.squeeze(0)

                    loss += criterion(decoder_output, sample_target_tensor[di].view(1))
                    num_values += 1

                    if decoder_input.item() == EOS_token:
                        break

            loss = loss/num_values # divide by mini-batch size

    loss.backward()
    optimizer.step()

    # Call detach to return a tensor detached from the current graph
    return loss.detach()


def trainIters(experiment_name, train_data, input_lang, output_lang, encoder, decoder, n_epochs, teacher_forcing_ratio, optimizer, simplify=False,
               reverse_input=False, use_attention=False, learning_rate=0.01, batch_size=2):
    # start = time.time()
    # plot_losses = []

    print("Training for {} epochs with batch size {}, learning rate {} and teacher forcing ratio {}".
          format(n_epochs, batch_size, learning_rate, teacher_forcing_ratio))
    print("Training with attention: {}".format(use_attention))

    if simplify:
        train_data = filterPairs(train_data)
        print("Simplifying training dataset to have {} examples".format(len(train_data)))

    encoder.train()
    decoder.train()

    criterion = nn.NLLLoss(ignore_index=PAD_token, reduction='none')

    for epoch_idx in range(n_epochs):

        print_loss_total = 0.
        plot_loss_total = 0.
        start_batch_idx = 0
        end_batch_idx = np.minimum(batch_size, len(train_data))
        num_minibatches = 0.

        while start_batch_idx < len(train_data):

            input_tensor, target_tensor = get_minibatch(train_data[start_batch_idx:end_batch_idx],
                                                        input_lang, output_lang, reverse_input)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, optimizer, criterion, teacher_forcing_ratio, use_attention)

            print_loss_total += loss.item()
            plot_loss_total += loss.item()

            start_batch_idx = end_batch_idx
            end_batch_idx = np.minimum(end_batch_idx + batch_size, len(train_data))
            num_minibatches += 1

            # if end_batch_idx % print_every == 0:
            #     print_loss_avg = print_loss_total / print_every
            #     print_loss_total = 0
            #     print('%s (%d %d%%) %.4f' % (timeSince(start, end_batch_idx / n_iters),
            #                                  iter, iter / n_iters * 100, print_loss_avg))
            #
            # if end_batch_idx % plot_every == 0:
            #     plot_loss_avg = plot_loss_total / plot_every
            #     plot_losses.append(plot_loss_avg)
            #     plot_loss_total = 0

        #showPlot(plot_losses)
        print("[{}] Average Loss after epoch {}/{}: {:5f}".format(experiment_name, epoch_idx + 1, n_epochs,
                                                                  print_loss_total/num_minibatches))
        save_checkpoint(epoch_idx + 1, encoder, decoder, optimizer, print_loss_total/num_minibatches,
                        filename=experiment_name + ".tar")


def evaluate(input_lang, output_lang, encoder, decoder, pair, reverse_input=False, use_attention=False):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        input_tensor, _ = get_minibatch(pair, input_lang, output_lang, reverse_input)

        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden(input_length)
        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        if not use_attention:
            encoder_output = None

        results = beam_search(decoder, encoder_hidden, encoder_output)

        idx_highest_prob = 0
        highest_prob = -1000.

        for idx, (prob, _) in enumerate(results):
            if prob.item() > highest_prob:
                idx_highest_prob = idx
                highest_prob = prob.item()

        output = [output_lang.index2word[idx] for idx in results[idx_highest_prob][1]]

        return output


def greedy_search(decoder, encoder_hidden, output_lang):
    decoded_words = []
    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

    # An EOS token has to be spit out eventually
    while True:
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)

        if topi.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            top_item = output_lang.index2word[topi.item()]
            decoded_words.append(top_item)

        decoder_input = topi.squeeze().detach()

    print([word for word in decoded_words])


def beam_search(decoder, encoder_hidden, encoder_output=None, beam_width=3):
    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

    # First step of decoding
    decoder_output, decoder_hidden = decoder(decoder_input, encoder_hidden, encoder_output)

    top_values, top_indices = decoder_output.data.topk(beam_width)
    probs = top_values.squeeze()
    top_indices = top_indices.squeeze()
    to_evaluate = [(decoder, decoder_hidden, probs[i], top_indices[i]) for i in range(beam_width)]

    candidates = [[] for _ in range(beam_width)]
    results = []

    # Go until all candidates have not seen the EOS
    while len(to_evaluate) > 0:
        # Mean to store an array of tensors of length 'beam_width'
        new_probs = []

        for idx, (decoder_, decoder_hidden_, prob_, top_idx_) in enumerate(to_evaluate):
            decoder_output, temp_decoder_hidden = decoder_(top_idx_, decoder_hidden_, encoder_output)
            to_evaluate[idx] = (decoder_, temp_decoder_hidden, prob_, top_idx_)
            new_probs.append(prob_ + decoder_output.squeeze())  # Compute new probabilities

        which_tensor, probs, indices = get_largest(new_probs, beam_width - len(results)) # beam is reduced as results come in

        # Refresh the candidates according to the highest probabilities
        # which_tensor tells us the models that gave us the highest probabilities
        old_candidates = candidates.copy()

        for idx, which_one in enumerate(which_tensor):
            candidates[idx] = old_candidates[which_one].copy()

        to_evaluate_copy = to_evaluate.copy()
        to_evaluate = []

        for idx_, (which_, value_, top_idx_) in enumerate(zip(which_tensor, probs, indices)):
            candidates[idx_].append(top_indices[which_].item())
            to_evaluate.append((to_evaluate_copy[which_][0], to_evaluate_copy[which_][1], value_, top_idx_))

        new_candidates = []
        to_evaluate_copy = []

        # If we encounter an EOS, this candidate is a possible result
        for idx_, to_evaluate_tuple in enumerate(to_evaluate):
            if to_evaluate_tuple[3] == EOS_token:
                results.append((to_evaluate_tuple[2], candidates[idx_]))  # Add probability and sequence
            else:
                new_candidates.append(candidates[idx_])
                to_evaluate_copy.append(to_evaluate_tuple)

        candidates = new_candidates
        to_evaluate = to_evaluate_copy
        top_indices = indices

    return results

            
def print_results(input_sentences, output_tensor, output_lang):

    for sample_idx in range(output_tensor.size(0)):

        decoded_words = []

        for token_idx in range(output_tensor.size(1)):

            topv, topi = output_tensor[sample_idx][token_idx].data.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

        print(input_sentences[sample_idx], '->', decoded_words)


def evaluateRandomly(valid_data, input_lang, output_lang, encoder, decoder,
                     simplify=False, n=10, reverse_input=False, use_attention=False):

    if simplify:
        valid_data = filterPairs(valid_data)

    for i in range(n):
        pair = random.choice(valid_data)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(input_lang, output_lang, encoder, decoder, [pair], reverse_input, use_attention)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def get_largest(tensors, num_largest):
    """
    Given an array of 1D tensors of equal size, return three arrays:
    1. Which tensors return the largest values
    2. Indices of largest values inside the respective tenssors
    3. Largest values
    :param tensors:
    :return:
    """
    # Get size of tensors, assume they are all of equal length
    tensor_len = tensors[0].size(0)
    # Concatenate tensors
    concatenated = torch.cat(tensors)
    # Call topk on tensors. This gives us 2/3 of the answer
    top_values, top_idx = concatenated.topk(num_largest)

    # Using the size of tensors, return which tensors had the largest values
    which = top_idx/tensor_len

    return which, top_values, top_idx % tensor_len


def save_checkpoint(epoch, encoder, decoder, optimizer, loss, filename="checkpoint.tar"):
    state = {
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }

    torch.save(state, filename)
    print("Saved checkpoint {}".format(filename))


def load_checkpoint(filename, encoder, decoder, optimizer):
    print("Loading checkpoint saved in {}".format(filename))

    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("Loaded checkpoint - epoch {} having loss {}".format(epoch, loss))

    return encoder, decoder, optimizer, epoch, loss


def get_datasets(from_lang, to_lang):
    """
    Return two datasets as lists, one for training and one for development
    :param from_lang: source language
    :param to_lang: target language
    :return:
    """
    train_file = "data/%s-%s.train" % (from_lang, to_lang)
    dev_file = "data/%s-%s.dev" % (from_lang, to_lang)

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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
