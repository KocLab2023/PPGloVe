from cooccurrence import *
from Paillier_algorithm import *
from evalute import *
from timeit import default_timer as timer
import numpy as np


def generate_encrypt_cooccur(corpus, pub, exponent=24, min_count=None):
    vocab = build_vocab(corpus)
    cooccurrences = build_cooccur1(vocab, corpus)
    id2word = dict((i, word) for word, (i, _) in vocab.items())

    for i in range(cooccurrences.shape[0]):
        for j in range(len(cooccurrences.data[i])):

            cooccurrences.data[i][j] = round(cooccurrences.data[i][j] * (2 ** exponent)).astype(np.int64)
            cooccurrences.data[i][j] = paillier_encrypt(pub, cooccurrences.data[i][j].item())      


    for i, (row, data) in enumerate(zip(cooccurrences.rows, cooccurrences.data)):

        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue

            yield i, j, data[data_idx]
    return cooccurrences


if __name__ == '__main__':
    priv, pub = paillier_generate_keypair(1024)

    corpus1, corpus2, corpus3, corpus4 = load_datasets()


    start1 = timer()
    enc_cooccurrences1 = generate_encrypt_cooccur(corpus1, pub)
    # for i_main, i_context, enc_cooccurrence in enc_cooccurrences1:
    #     break
    #     print([i_main, i_context, enc_cooccurrence])
    end1 = timer()
    time1 = end1 - start1
    print(f"Encrypted time1 is {time1: 0.3f} s")
    np.save(file="enc_cooccurrences1.npy", arr=enc_cooccurrences1)


    start2 = timer()
    enc_cooccurrences2 = generate_encrypt_cooccur(corpus2, pub)
    # for i_main, i_context, enc_cooccurrence in enc_cooccurrences2:
    #     break
    #     print([i_main, i_context, enc_cooccurrence])
    end2 = timer()
    time2 = end2 - start2
    print(f"Encrypted time2 is {time2: 0.3f} s")
    np.save(file="enc_cooccurrences2.npy", arr=enc_cooccurrences2)


    start3 = timer()
    enc_cooccurrences3 = generate_encrypt_cooccur(corpus3, pub)
    # for i_main, i_context, enc_cooccurrence in enc_cooccurrences3:
    #     break
    #     print([i_main, i_context, enc_cooccurrence])
    end3 = timer()
    time3 = end3 - start3
    print(f"Encrypted time3 is {time3: 0.3f} s")
    np.save(file="enc_cooccurrences3.npy", arr=enc_cooccurrences3)


    start4 = timer()
    enc_cooccurrences4 = generate_encrypt_cooccur(corpus4, pub)
    # for i_main, i_context, enc_cooccurrence in enc_cooccurrences4:
    #     break
    #     print([i_main, i_context, enc_cooccurrence])
    end4 = timer()
    time4 = end4 - start4
    print(f"Encrypted time4 is {time4: 0.3f} s")
    np.save(file="enc_cooccurrences4.npy", arr=enc_cooccurrences4)
