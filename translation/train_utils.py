import torch
import random, time
from util import indexesFromSentence, timeSince, showPlot
from torch import optim
from lang import SOS_token, EOS_token
from torch.nn.utils.rnn import pack_sequence
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device)#.view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_minibatch(pairs, input_lang, output_lang):
    """
    :param pairs: list of lists, the inner lists have two elements, the first one contains the input sentence and the
                second one contains the output sentences
    :return: two tensors: one for the source language and the other for the target language. The first dimension of
            each tensor is equal to the length of the input list. The second dimension depends on the maximum number of
            tokens in the input and output sentences
    """
    max_len_input = 0
    max_len_output = 0
    list_input_sentences = []
    list_output_sentences = []

    # Go through each pair and keep track of largest sentence to adjust the padding at the end
    for pair in pairs:

        input_tensor = tensorFromSentence(input_lang, pair[0])
        output_tensor = tensorFromSentence(output_lang, pair[1])

        if input_tensor.size(0) > max_len_input:
            max_len_input = input_tensor.size(0)

        if output_tensor.size(0) > max_len_output:
            max_len_output = output_tensor.size(0)

        list_input_sentences.append(input_tensor)
        list_output_sentences.append(output_tensor)

    # To use the PyTorch's pack_sequence function, we need to make sure the list is in decreasing order in terms of len
    if max_len_input > list_input_sentences[0].size(0):
        replacement_input = torch.zeros(max_len_input) # create tensor of correct length
        replacement_input[:list_input_sentences[0].size(0)] = list_input_sentences[0]  # fill the new tensor with data
        list_input_sentences[0] = replacement_input #replace the first tensor

    print(max_len_output, list_output_sentences[3].size(0))

    #TODO: need to make the list in ascending order, not just the first element
    if max_len_output > list_output_sentences[0].size(0):
        replacement_output = torch.zeros(max_len_output)
        replacement_output[:list_output_sentences[0].size(0)] = list_output_sentences[0]
        print(replacement_output)
        list_output_sentences[0] = replacement_output

    return pack_sequence(list_input_sentences)#, pack_sequence(list_output_sentences)



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, SOS_token, EOS_token, teacher_forcing_ratio, batch_size):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # which dimension tells me the size of the sequence?
    encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1) #Get the most probable word
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(pairs, input_lang, output_lang, encoder, decoder, n_iters,
               print_every=1000, plot_every=100, learning_rate=0.01, batch_size=1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    print(get_minibatch(pairs[:5], input_lang, output_lang))

    # Each element of the pair is an array of long tensors
    # training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang)
    #                   for i in range(n_iters)]
    #
    # print("Training {} pairs".format(len(training_pairs)))
    #
    # criterion = nn.NLLLoss()
    #
    # #Assuming length of training pairs is larger or equal than batch size
    # start_batch = 0
    # end_batch = batch_size
    #
    # #for iter in range(1, n_iters + 1):
    # while end_batch < len(training_pairs):
    #
    #     training_pair = training_pairs[start_batch: end_batch]
    #
    #     print(training_pair)
    #
    #     start_batch= end_batch
    #     end_batch += batch_size
    #
    # #     input_tensor = training_pair[0]
    # #     target_tensor = training_pair[1]
    # #
    # #     loss = train(input_tensor, target_tensor, encoder,
    # #                  decoder, encoder_optimizer, decoder_optimizer, criterion, SOS_token, EOS_token, 0.5, 1)
    # #     print_loss_total += loss
    # #     plot_loss_total += loss
    # #
    # #     if iter % print_every == 0:
    # #         print_loss_avg = print_loss_total / print_every
    # #         print_loss_total = 0
    # #         print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
    # #                                      iter, iter / n_iters * 100, print_loss_avg))
    # #
    # #     if iter % plot_every == 0:
    # #         plot_loss_avg = plot_loss_total / plot_every
    # #         plot_losses.append(plot_loss_avg)
    # #         plot_loss_total = 0
    # #
    # # showPlot(plot_losses)

def evaluate(input_lang, output_lang, encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(input_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluateRandomly(pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')