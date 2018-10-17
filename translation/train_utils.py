import torch
import random, time
from util import indexesFromSentence, timeSince, showPlot
from torch import optim
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


def get_minibatch(pairs, input_lang, output_lang):
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
        output_tensor = tensorFromSentence(output_lang, pair[1], True)
        list_input_sentences.append(input_tensor)
        list_output_sentences.append(output_tensor)

    return pad_sequence(list_input_sentences, batch_first=True), pad_sequence(list_output_sentences, batch_first=True)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, teacher_forcing_ratio):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(1)

    encoder_hidden = encoder.initHidden(input_length)

    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    loss = 0

    #TODO how do we handle EOS token if doing entire sequence all at once?
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input, ignore <EOS> token at the end
        decoder_output, decoder_hidden = decoder(target_tensor[:, :-1], decoder_hidden)

        # Need to swap the dimensions not corresponding to the minibatch for NLLoss to work
        # To calculate loss, ignore <SOS> token
        loss_batch = criterion(decoder_output.transpose(1, 2), target_tensor[:, 1:])

        # Remember that we padded the target tensor, so find out the number of outputs
        # Sum the loss and divide by the number of outputs
        loss = loss_batch.sum() / torch.nonzero(target_tensor[:, 1:]).size(0)

        #print("teacher_forcing loss", loss.data)
    else:
        # Without teacher forcing: use its own predictions as the next input
        # encoder_hidden: 1 x batch_size x d_hidden
        # For each hidden state in the minibatch
        # For as many tokens in the target until finding EOS, pass the previous output
        # In this case, we don't set decoder_hidden = encoder_hidden because we use decoder_hidden below
        num_values = 0

        for sample_idx in range(encoder_hidden.size(1)):
            #print(decoder_hidden[0][sample_idx].size())
            decoder_hidden = encoder_hidden[0][sample_idx].view(1, 1, -1)
            decoder_input = torch.tensor([[SOS_token]], device=device)

            sample_target_tensor = target_tensor[sample_idx, 1:]
            #print(sample_target_tensor.size())

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

    encoder_optimizer.step()
    decoder_optimizer.step()

    # Call detach to return a tensor detached from the current graph
    return loss.detach()


def trainIters(train_data, input_lang, output_lang, encoder, decoder, n_epochs, learning_rate=0.01, batch_size=2):
    start = time.time()
    plot_losses = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_token, reduction='none')

    for epoch_idx in range(n_epochs):

        print_loss_total = 0.
        plot_loss_total = 0.
        start_batch_idx = 0
        end_batch_idx = np.minimum(batch_size, len(train_data))
        num_minibatches = 0.

        while start_batch_idx < len(train_data):

            input_tensor, target_tensor = get_minibatch(train_data[start_batch_idx:end_batch_idx], input_lang, output_lang)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, 0.5)

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
        print("Average Loss after epoch {}/{}: {:5f}".format(epoch_idx + 1, n_epochs, print_loss_total/num_minibatches))
        save_checkpoint(epoch_idx + 1, encoder, decoder, encoder_optimizer, decoder_optimizer, print_loss_total/num_minibatches)


def evaluate(input_lang, output_lang, encoder, decoder, pair):
    with torch.no_grad():

        input_tensor, _ = get_minibatch(pair, input_lang, output_lang)

        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden(input_length)
        _, encoder_hidden = encoder(input_tensor, encoder_hidden)

        #greedy_search(decoder, encoder_hidden, output_lang)

        results = beam_search(decoder, encoder_hidden)

        for i in range(len(results)):
            print([output_lang.index2word[idx] for idx in results[i][0]], results[i][1].item())

        return []


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


def beam_search(decoder, encoder_hidden, beam_width=3):
    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

    # First step of decoding
    decoder_output, decoder_hidden = decoder(decoder_input, encoder_hidden)

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
            decoder_output, temp_decoder_hidden = decoder_(top_idx_, decoder_hidden_)
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
                results.append((candidates[idx_], to_evaluate_tuple[2]))  # Add sequence and likelihood
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


def evaluateRandomly(valid_data, input_lang, output_lang, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(valid_data)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(input_lang, output_lang, encoder, decoder, [pair])
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


def save_checkpoint(epoch, encoder, decoder, enc_optim, dec_optim, loss, filename="checkpoint.tar"):
    state = {
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'enc_optim': enc_optim.state_dict(),
        'dec_optim': dec_optim.state_dict(),
        'loss': loss
    }

    torch.save(state, filename)


def load_checkpoint(filename, encoder, decoder, enc_optim, dec_optim):
    print("Loading checkpoint saved in {}".format(filename))

    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    enc_optim.load_state_dict(checkpoint['enc_optim'])
    dec_optim.load_state_dict(checkpoint['dec_optim'])

    print("Loaded checkpoint - epoch {} having loss {}".format(epoch, loss))

    return encoder, decoder, enc_optim, dec_optim, epoch, loss


# def evaluate(input_lang, output_lang, encoder, decoder, pair):
#     with torch.no_grad():
#
#         # start_batch_idx = 0
#         # end_batch_idx = np.minimum(batch_size, len(valid_data))
#
#         # while start_batch_idx < len(valid_data):
#
#         input_tensor, target_tensor = get_minibatch(pair, input_lang, output_lang)
#
#         input_length = input_tensor.size(0)
#         encoder_hidden = encoder.initHidden(input_length)
#         encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)
#         encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
#
#         # TODO change this to use beam search
#
#         decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
#         decoder_hidden = encoder_hidden
#         decoded_words = []
#         beam_width = 4
#
#         # First step of decoding
#         decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
#
#         top_values, top_indices = decoder_output.data.topk(beam_width)
#         probs = top_values.squeeze()
#         top_indices = top_indices.squeeze()
#         decoders = [decoder for _ in range(beam_width)]
#         decoders_hidden = [decoder_hidden for _ in range(beam_width)]
#
#         # For candidates
#         candidates = [[] for _ in range(beam_width)]
#         candidates_probs = []
#
#         # An EOS token has to be spit out eventually
#         while len(top_indices) > 0:
#             # Mean to store an array of tensors of length 'beam_width'
#             new_probs = []
#
#             for idx, token_idx in enumerate(top_indices):
#                 decoder_output, decoders_hidden[idx] = decoders[idx](token_idx, decoders_hidden[idx])
#                 new_probs.append(probs[idx] + decoder_output.squeeze()) # Compute new probabilities
#
#             #print(new_probs)
#             which_tensor, values, indices = get_largest(new_probs, beam_width)
#
#             # Need to populate decoders and decoders_hidden, as well as probs, and keep track of candidates
#             print(which_tensor, values, indices)
#
#             # Reset the candidates so far
#             for idx, which_one in enumerate(which_tensor):
#                 # Make sure the past is that of the candidate
#                 candidates[idx] = candidates[which_one].copy()
#
#             temp_decoders = decoders.copy()
#             temp_decoders_hidden = decoders_hidden.copy()
#
#             for idx, which_one in enumerate(which_tensor):
#                 # Store the token that led to this high probability
#                 candidates[idx].append(top_indices[which_one].item())
#
#                 if indices[idx] != EOS_token: # we keep going
#                     decoders[idx] = temp_decoders[which_one]
#                     decoders_hidden[idx] = temp_decoders_hidden[which_one]
#                     probs[idx] = values[idx]
#                 else:
#                     import sys
#                     sys.exit(1)
#
#             for i in range(len(candidates)):
#                 print([output_lang.index2word[idx] for idx in candidates[i]])
#
#             top_indices = indices
#
#         return candidates