from encrypt_cooccur import *
from Paillier_algorithm import *
from gmpy2 import random_state, mpz_urandomb, invert


rand = random_state(random.randrange(sys.maxsize))

def secure_multi(cipher1, cipher2, pub, priv, bits=16):
    r1 = mpz_urandomb(rand, bits)
    r2 = mpz_urandomb(rand, bits)
    cipher_r1 = paillier_encrypt(pub, r1)
    cipher_r2 = paillier_encrypt(pub, r2)

    cipher_d1 = cipher1 * cipher_r1 % pub.n_sq
    cipher_d2 = cipher2 * cipher_r2 % pub.n_sq

    d1 = paillier_decrypt(priv, pub, cipher_d1)
    d2 = paillier_decrypt(priv, pub, cipher_d2)

    d = d1 * d2           

    cipher_d = paillier_encrypt(pub, d)

    cipher_r1r2 = paillier_encrypt(pub, r1 * r2)
    multi_result = cipher_d * invert(cipher1 ** r2, pub.n_sq) * invert(cipher2 ** r1, pub.n_sq) * invert(cipher_r1r2, pub.n_sq) % pub.n_sq

    return multi_result

def secure_comparison(cipher1, cipher2, pub, priv, bits=16):
    while True:
        r1 = mpz_urandomb(rand, bits)
        r2 = mpz_urandomb(rand, bits)
        r3 = mpz_urandomb(rand, bits)
        if (r3 < r2) and (r3 < r1):
            break
    t1 = random.randint(0, 1)
    if t1 == 0:
        cipher_s = r3 * ((cipher1 * invert(cipher2, pub.n_sq)) ** r1) % pub.n_sq
    else:
        cipher_s = r3 * ((invert(cipher1, pub.n_sq) * cipher2) ** r2) % pub.n_sq

    s = paillier_decrypt(priv, pub, cipher_s)

    if s < pub.n / 2:
        t2 = 0
    else:
        t2 = 1

    if t2 == t1:
        return 0
    else:
        return 1

def secure_compute_weight(cooccur, pub, priv, exponent=24):
    x_max = 100
    alpha = 1

    cipher_x_max = paillier_encrypt(pub, x_max)
    inverse_cipher_x_max = invert(cipher_x_max, pub.n_sq)


    compar_result = secure_comparison(cooccur, cipher_x_max, pub, priv)
    if compar_result == 1:
        weight = secure_multi(cooccur, inverse_cipher_x_max, pub, priv)
    else:
        weight = paillier_encrypt(pub, 1)

    return weight


def secure_vector_inner_product(cipher_vec1, cipher_vec2, pub, priv):
    len_vec = len(cipher_vec1)
    cipher_product = 1

    for i in range(len_vec):

        cipher_multi = secure_multi(cipher_vec1[i].astype(np.int64).item(), cipher_vec2[i].astype(np.int64).item(), pub, priv)
        cipher_product = (cipher_product * cipher_multi) % pub.n_sq

    return cipher_product











