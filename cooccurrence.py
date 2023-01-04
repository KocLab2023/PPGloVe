from collections import Counter
from scipy import sparse
from data_preprocces import *
from timeit import default_timer as timer
import numpy as np

import itertools

def build_vocab(corpus):
    """
    Build a vocabulary with word frequencies for an entire corpus.
    """

    print("Building vocab from corpus")

    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}

def build_cooccur(vocab, corpus, window_size=100, min_count=None):
    """
    Build a word co-occurrence list for the given corpus.
    """

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.items())

    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)

    for line in corpus:
        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]


        for center_i, center_id in enumerate(token_ids):

            context_ids = token_ids[max(0, center_i - window_size): center_i]

            context_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):

                distance = context_len - left_i

                increment = 1.0 / float(distance)

                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    for i, (row, data) in enumerate(zip(cooccurrences.rows, cooccurrences.data)):

        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue

            yield i, j, data[data_idx]
    return cooccurrences

def build_cooccur1(vocab, corpus, window_size=100, min_count=None):

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.items())

    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)

    for line in corpus:
        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]


        for center_i, center_id in enumerate(token_ids):

            context_ids = token_ids[max(0, center_i - window_size): center_i]

            context_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):

                distance = context_len - left_i

                increment = 1.0 / float(distance)

                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    return cooccurrences


