import io
from util import normalizeString
from tqdm import tqdm
import argparse
from collections import Counter

###############################################################################################
# Need to tokenize according to given word embeddings, so do a quick search that tokenization makes sense
# - don't -> don 't OR don ' t
#
###############################################################################################

# sents is a list of sentences
# max_tokens indicates the maximum number of tokens in a sentence
# Return processed sentences and vocabulary Counter
def process_sents(sents, max_tokens):
    cleaned = []

    vocab = Counter()

    for line in tqdm(sents):
        # normalize unicode characters
        line = normalizeString(line)

        if len(line.split()) <= max_tokens:
            # store as string
            cleaned.append(line)
            # Update vocabulary counter
            vocab.update(line.split())

    return cleaned, vocab


# corpus is a bilingual corpus separated by a tab token
def extract_vocab(corpus, max_tokens, size_vocab):

    from_sents = []
    to_sents = []

    with open(corpus, mode='rt', encoding='utf-8')as file:
        corpus_content = file.readlines()

    print("Creating vocabularies")

    for line in tqdm(corpus_content):
        from_sent, to_sent = line.split('\t')
        from_sents.append(from_sent)
        to_sents.append(to_sent)

    from_sents, from_vocab = process_sents(from_sents, max_tokens)
    to_sents, to_vocab = process_sents(to_sents, max_tokens)

    print("Found {} FROM sentences".format(len(from_sents)))
    print("Found {} TO sentences".format(len(to_sents)))

    # Only keep the desired number of tokens in vocabulary
    most_common_from = from_vocab.most_common(size_vocab)
    from_vocab = [k for k, c in most_common_from]

    most_common_to = to_vocab.most_common(size_vocab)
    to_vocab = [k for k, c in most_common_to]

    return from_vocab, to_vocab


def write_vocab_emb(vocab, embs, vocab_emb_file):
    emb_handler = io.open(embs, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, emb_handler.readline().split())
    print("Found {} embeddings with dimension {}".format(n, d))

    data = {}

    for line in emb_handler:
        tokens = line.rstrip().split(' ')

        if tokens[0] in vocab:
            data[tokens[0]] = tokens[1:]
            # already found token so remove it from vocab
            vocab.remove(tokens[0])

        # Once we cover all vocabulary, exit
        if not vocab:
            break

    with open(vocab_emb_file, 'w') as file:
        file.write("{} {}\n".format(len(data), d))

        for k, v in data.items():
            file.write("{} {}\n".format(k, " ".join(v)))

    print("Finished writing embeddings containing {} tokens".format(len(data)))


def main():
    parser = argparse.ArgumentParser(description="Given a corpus, create vocabularies for both languages")
    parser.add_argument("corpus", help="Corpus", type=str)
    parser.add_argument("from_emb", help="Word embeddings for source language", type=str)
    parser.add_argument("to_emb", help="Word embeddings for target language", type=str)
    parser.add_argument("from_vocab", help="File where to output the from vocabulary", type=str)
    parser.add_argument("to_vocab", help="File where to output the to vocabulary", type=str)
    parser.add_argument("--max_tokens", help="Maximum number of tokens per sentence", type=int, default=50)
    parser.add_argument("--size_vocab", help="Size of vocabulary to create", type=int, default=50000)

    args = parser.parse_args()

    corpus = args.corpus
    from_vocab_file = args.from_vocab
    to_vocab_file = args.to_vocab
    from_emb = args.from_emb
    to_emb = args.to_emb
    max_tokens = args.max_tokens
    size_vocab = args.size_vocab

    from_vocab_list, to_vocab_list = extract_vocab(corpus, max_tokens, size_vocab)

    write_vocab_emb(from_vocab_list, from_emb, from_vocab_file)
    write_vocab_emb(to_vocab_list, to_emb, to_vocab_file)


if __name__ == "__main__":
    main()
