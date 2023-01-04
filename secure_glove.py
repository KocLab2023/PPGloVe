from secure_basic import *
from random import shuffle
from gmpy2 import *

def secure_train_glove(vocab, cooccurrences, vector_size = 100, iterations = 25, exponent=24, **kwargs):
    vocab_size = len(vocab)

    W = ((np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1))
    biases = ((np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1))


    for i in range(len(biases)):
        biases[i] = round(biases[i] * (2 ** exponent))
        if biases[i] < 0:
            biases[i] += 2 ** exponent
        for j in range(W.shape[1]):
            W[i][j] = round(W[i][j] * (2 ** exponent))
            if W[i][j] < 0:
                W[i][j] += 2 ** exponent

    data = [(W[i_main],
             W[i_context + vocab_size],
             biases[i_main: i_main + 1],
             biases[i_context + i_context: i_context + vocab_size + 1],
             cooccurrence)
            for i_main, i_context, cooccurrence in cooccurrences]

    for (v_main, v_context, b_main, b_context, cooccurrence) in data:
        for i in range(vector_size):
            v_main[i] = paillier_encrypt(pub, v_main[i].astype(np.int64).item())
            v_context[i] = paillier_encrypt(pub, v_context[i].astype(np.int64).item())
        for i in range(len(b_main)):
            b_main[i] = paillier_encrypt(pub, b_main[i].astype(np.int64).item())
        for i in range(len(b_context)):
            b_context[i] = paillier_encrypt(pub, b_context[i].astype(np.int64).item())


    for i in range(iterations):
        print(f"Beginning iteration {i}...")

        secure_run_iter(data, pub, priv, **kwargs)

    return W

def secure_run_iter(data, pub, priv, learning_rate = 0.05, exponent=24):
    global_cost = 0
    shuffle(data)

    for (cipher_v_main, cipher_v_context, cipher_b_main, cipher_b_context, cooccurrence) in data:
        weight = secure_compute_weight(cooccurrence, pub, priv)

        cipher_log = ((cooccurrence * invert(secure_multi(cooccurrence, cooccurrence, pub, priv) * round(0.5 * (2 ** exponent)) % pub.n_sq, pub.n_sq)) % pub.n_sq \
                     * (secure_multi(cooccurrence, cooccurrence, pub, priv) * round(1 / 3 * (2 ** exponent))) % pub.n_sq) % pub.n_sq

        cipher_cost_inner = (((secure_vector_inner_product(cipher_v_main, cipher_v_context, pub, priv) // (2 ** exponent) \
                            * cipher_b_main[0]) % pub.n_sq * cipher_b_context[0]) % pub.n_sq * cipher_log) % pub.n_sq

        # for i in range(len(cipher_cost_inner)):
        # cipher_cost = secure_multi(secure_compute_weight(cooccurrence, pub, priv), secure_multi(cipher_cost_inner, cipher_cost_inner, pub, priv))


        cipher_grad_main = np.ones(len(cipher_v_main))
        cipher_grad_context = np.ones(len(cipher_v_context))
        for i in range(len(cipher_v_context)):
            cipher_grad_main[i] = secure_multi(secure_multi(weight, mpz(cipher_cost_inner), pub, priv), cipher_v_context[i].astype(np.int64).item(), pub, priv)
            cipher_grad_context[i] = secure_multi(secure_multi(weight, mpz(cipher_cost_inner), pub, priv), cipher_v_main[i].astype(np.int64).item(), pub, priv)

        cipher_grad_bias_main = secure_multi(weight, mpz(cipher_cost_inner), pub, priv)
        cipher_grad_bias_context = secure_multi(weight, mpz(cipher_cost_inner), pub, priv)


        for i in range(len(cipher_grad_main)):
            cipher_v_main[i] = (cipher_v_main[i] * invert(cipher_grad_main[i].astype(np.int64).item() ** round(learning_rate * (2 ** exponent)) % pub.n_sq, pub.n_sq)) % pub.n_sq
            cipher_v_context[i] = (cipher_v_context[i] * invert(cipher_grad_context[i].astype(np.int64).item() ** round(learning_rate * (2 ** exponent)) % pub.n_sq, pub.n_sq)) % pub.n_sq
        print(cipher_v_main)

        cipher_b_main = (cipher_b_main * invert(cipher_grad_bias_main, pub.n_sq)) % pub.n_sq
        cipher_b_context = (cipher_b_context * invert(cipher_grad_bias_context, pub.n_sq)) % pub.n_sq


if __name__ == '__main__':
    priv, pub = paillier_generate_keypair(1024)


    corpus1, corpus2, corpus3, corpus4 = load_datasets()

    vocab1 = build_vocab(corpus1)
    enc_cooccurrences1 = np.load(file="enc_cooccurrences1.npy", allow_pickle=True)
    enc_cooccurrences1 = transfrom_data(corpus1, enc_cooccurrences1)
    start = timer()
    W1 = secure_train_glove(vocab1, enc_cooccurrences1, vector_size = 100, iterations = 25)
    end = timer()
    time1 = end - start
    print(f"Running time1 is {time1: 0.3f} s")
    np.save(file="W1.npy", arr=W1)
    #
    # vocab2 = build_vocab(corpus2)
    # enc_cooccurrences2 = np.load(file="enc_cooccurrences2.npy", allow_pickle=True)
    # enc_cooccurrences2 = transfrom_data(corpus2, enc_cooccurrences2)
    # start = timer()
    # W2 = secure_train_glove(vocab2, enc_cooccurrences2, vector_size = 100, iterations = 25)
    # end = timer()
    # time2 = end - start
    # print(f"Running time2 is {time2: 0.3f} s")
    #
    # vocab3 = build_vocab(corpus3)
    # enc_cooccurrences3 = np.load(file="enc_cooccurrences3.npy", allow_pickle=True)
    # enc_cooccurrences3 = transfrom_data(corpus3, enc_cooccurrences3)
    # start = timer()
    # W3 = secure_train_glove(vocab3, enc_cooccurrences3, vector_size = 100, iterations = 25)
    # end = timer()
    # time3 = end - start
    # print(f"Running time3 is {time3: 0.3f} s")
    #
    # vocab4 = build_vocab(corpus4)
    # enc_cooccurrences4 = np.load(file="enc_cooccurrences4.npy", allow_pickle=True)
    # enc_cooccurrences4 = transfrom_data(corpus4, enc_cooccurrences4)
    # start = timer()
    # W4 = secure_train_glove(vocab4, enc_cooccurrences4, vector_size = 100, iterations = 25)
    # end = timer()
    # time4 = end - start
    # print(f"Running time4 is {time4: 0.3f} s")

    # W_dec = [[1] * W.shape[1] for _ in range(W.shape[0])]
    # for i in range(W.shape[0]):
    #     for j in range(W.shape[1]):
    #         W_dec[i][j] = W[i][j]
    #
    #         W_dec[i][j] = paillier_decrypt(priv, pub, mpz(W_dec[i][j].item()))
    #
    # print("---------")
    # print(W_dec)

