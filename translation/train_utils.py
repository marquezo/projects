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

    encoder_hidden = encoder.initHidden(input_length)

    encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True #if random.random() < teacher_forcing_ratio else False

    #TODO how do we handle EOS token if doing entire sequence all at once?
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input, ignore <EOS> token at the end
        decoder_output, decoder_hidden = decoder(target_tensor[:, :-1], decoder_hidden)

        # Need to swap the dimensions not corresponding to the minibatch for NLLoss to work
        # To calculate loss, ignore <SOS> token
        loss_batch = criterion(decoder_output.transpose(1, 2), target_tensor[:, 1:])

        # Remember that we padded the target tensor, so find out the number of outputs
        # Sum the loss and divide by the number of outputs
        loss = loss_batch.sum() / torch.nonzero(target_tensor).size(0)
    else:
        # Without teacher forcing: use its own predictions as the next input
        # decoder_hidden: 1 x batch_size x d_hidden

        # For each hidden state in the minibatch
        # The first token in the target is already SOS and we have already put an EOS
        # For as many tokens in the target until finding EOS, pass the previous output

        for sample_idx in range(decoder_hidden.size(1)):
            print(decoder_hidden[0][sample_idx].size())
        #
        # for di in range(target_length):
        #     decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        #     topv, topi = decoder_output.topk(1) #Get the most probable word
        #     decoder_input = topi.squeeze().detach()  # detach from history as input
        #
        #     loss += criterion(decoder_output, target_tensor[di])
        #
        #     if decoder_input.item() == EOS_token:
        #         break

        import sys
        sys.exit(1)



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

        print_loss_total = 0
        plot_loss_total = 0
        start_batch_idx = 0
        end_batch_idx = np.minimum(batch_size, len(train_data))

        while start_batch_idx < len(train_data):

            input_tensor, target_tensor = get_minibatch(train_data[start_batch_idx:end_batch_idx], input_lang, output_lang)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, 0.5)

            print_loss_total += loss.data
            plot_loss_total += loss

            #print(end_batch_idx, loss)

            start_batch_idx = end_batch_idx
            end_batch_idx = np.minimum(end_batch_idx + batch_size, len(train_data))

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
        print("Average Loss after epoch {}/{}: {:5f}".format(epoch_idx + 1, n_epochs, print_loss_total/len(train_data)))


def evaluate(input_lang, output_lang, encoder, decoder, pair):
    with torch.no_grad():

        # start_batch_idx = 0
        # end_batch_idx = np.minimum(batch_size, len(valid_data))

        # while start_batch_idx < len(valid_data):

        input_tensor, target_tensor = get_minibatch(pair, input_lang, output_lang)

        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden(input_length)
        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)
        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        # TODO change this to use beam search

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []

        # An EOS token has to be spit out eventually
        while True:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            print(decoder_output.data)
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                print(topv, topi)
                top_item = output_lang.index2word[topi.item()]
                print(top_item)
                decoded_words.append(top_item)

            decoder_input = topi.squeeze().detach()

        return decoded_words

            # start_batch_idx = end_batch_idx
            # end_batch_idx = np.minimum(end_batch_idx + batch_size, len(valid_data))

            
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
