import random, sys
from prime import *
from gmpy2 import mpz, powmod, random_state, mpz_random, gcd, invert

# generate random state
rand = random_state(random.randrange(sys.maxsize))

class PrivateKey(object):
    '''
    Contains a private key.

    Attributes:
        sk(mpz): private key
    '''
    def __init__(self, sk, p, q):
        self.sk = sk
        self.p = p
        self.q = q

class PublicKey(object):
    '''
    Contains a public key.

    Attributes:
        n(mpz): part of the public key
        g(mpz): part of thr public key
        n_sq(mpz): n ** 2, stored for frequent use
    '''
    def __init__(self, n, g):
        self.n = n
        self.n_sq = n ** 2
        self.g = g

class Data(object):
    '''
    Data related operations
    '''
    list_data = []

    def __init__(self, data):
        if type(data) in [list, tuple]:
            self.list_data = [int(i) if int(i) == i else i for i in data]
        elif type(data) == Data:
            self.list_data = data.get_values()

    def get_values(self):
        return self.list_data

    def __iter__(self):
        return iter(self.list_data)

    def __repr__(self):
        return repr(tuple(self.list_data))

    def __mul__(self, k):
        try:
            assert type(k) in [float, int]
        except:
            raise ValueError("c must be a scalar type")
        return Data([k * i for i in self.list_data])

    def __rmul__(self, k):
        try:
            assert type(k) in [float, int]
        except:
            raise ValueError("c must be a scalar type")
        return Data([k * i for i in self.list_data])

    def __iadd__(self, data):
        return Data([data.get_values()[i] + self.list_data[i] for i in range(len(self.list_data))])

    def __add__(self, data):
        return Data([data.get_values()[i] + self.list_data[i] for i in range(len(self.list_data))])

    def add_constant(self, k):
        return Data([k + i for i in self.list_data])

def paillier_generate_keypair(bits):
    '''
    Key generation algorithm for the Paillier cryptosystem.

    Params:
        bits(int): the size of prime

    Returns:
        tuple: generate keypair of class PrivateKey and class PublicKey.
    '''
    while True:
        p = generate_prime(bits // 2)
        q = generate_prime(bits // 2)
        if (p != q) and gcd(p, q - 1) == 1 and gcd(q, p - 1) == 1:
            break
    n = p * q
    # n_sq = n ** 2
    g = n + 1
    sk = (p - 1) * (q - 1)                        # Calculates secret key
    # while True:
    #     g = mpz_random(rand, n_sq)              # Chooses encryption base
    #     g1 = 1                                  # Initializes an element
    #     i = 0                                   # Initializes an element
    #     while True:
    #         g1 = (g1 * g) % n_sq
    #         i += 1
    #         if g1 == 1:                         # Calculates ord(g) in Zn2*
    #             break
    #     r = i % n                               # Checks whether ord(g) divides n
    #     if r == 0 and gcd(g, n_sq) == 1:
    #         break

    return PrivateKey(sk, p, q), PublicKey(n, g)

def paillier_encrypt(pub, plain):
    '''
     Encryption algorithm for the Paillier cryptosystem

    Params:
        pub(object): public key
        plain(int): plaintext of positive integer

    Return:
        ciphertext
    '''
    while True:
        u = mpz_random(rand, pub.n)
        if gcd(u, pub.n) == 1:
            break
    cipher = (powmod(pub.g, plain, pub.n_sq) * powmod(u, pub.n, pub.n_sq)) % pub.n_sq

    return cipher

def paillier_decrypt(priv, pub, cipher):
    '''
     Decryption algorithm for the Paillier cryptosystem

    Params:
        priv(object): private key
        pub(object): public key
        cipher(mpz): ciphertext

    Return:
        Recover plaintext
    '''
    z1 = powmod(cipher, priv.sk, pub.n_sq)
    u1 = (z1 - 1) // pub.n
    z2 = powmod(pub.g, priv.sk, pub.n_sq)
    u2 = invert((z2 - 1) // pub.n, pub.n)
    plain_cipher = (u1 * u2) % pub.n

    # x = powmod(cipher, priv.sk, pub.n_sq) - 1
    # plain_cipher = ((x // pub.n) * invert(priv.sk, pub.n)) % pub.n
    return plain_cipher

def paillier_homomorphic_additive(pub, cipher1, cipher2):
    '''
     Additively homomorphic for the Paillier cryptosystem
     D(E(m1) * E(m2)) = D(E(m1 + m2))

    Params:
        pub(object): public key
        cipher1(mpz): ciphertext of positive integer
        cipher2(mpz): ciphertext of positive integer

    Returns:
        E(m1) * E(m2)
    '''
    return cipher1 * cipher2 % pub.n_sq

def paillier_homomorphic_multiplicative_constant(pub, cipher, k):
    '''
     Multiplication by a constant for the Paillier cryptosystem
     D(E(m) ** k) = D(E(m) * k)

    Params:
        pub(object): public key
        cipher(mpz): ciphertext of positive integer
        k(int): a constant

    Returns:
        E(m) ** k
    '''
    return powmod(cipher, k, pub.n_sq)

def paillier_homomorphic_additive_constant(pub, cipher, k):
    '''
     Addition with a constant for the Paillier cryptosystem
     D(E(m) * (g ** k)) = D(E(k + m))

    Params:
        pub(object): public key
        cipher(mpz): ciphertext of positive integer
        k(int): a constant

    Returns:
        E(m) * (g ** k)
    '''
    return cipher * powmod(pub.g, k, pub.n_sq) % pub.n_sq

if __name__ == "__main__":
    # bits = 128
    # while True:
    #     p = generate_prime(bits)
    #     q = generate_prime(bits)
    #     if (p != q) and gcd(p, q - 1) == 1 and gcd(q, p - 1) == 1:
    #         break
    # print(p, q)
    p = 7
    q = 5
    priv, pub = paillier_generate_keypair(p, q)

    # print(priv.sk)
    # print(pub.n)
    # print(pub.g)

    # p, q = 13, 11
    # sk = 120
    # n, g = 143, 898
    # priv = PrivateKey(sk)
    # pub = PublicKey(n, g)
    #
    # plain = 24
    # cipher = encrypt(pub, plain)
    # plain_cipher = decrypt(priv, pub, cipher)
    # print(plain_cipher)

    # plain1, plain2 = add_homomorphic(pub, priv, 24, 18)
    # print(plain1, plain2)

    # plain1, plain2 = multi_integer_scalar(pub, priv, 24, 3)
    # print(plain1, plain2)

    # plain1, plain2 = add_integer_scalar(pub, priv, 24, 3)
    # print(plain1, plain2)

    # key_parameter = {"Prime p":p,
    #                  "Prime q":q,
    #                  "Private Key":priv,
    #                  "Public Key":pub}

    # write object to file
    # data = open("bits128.pkl", "wb")
    # pickle.dump(key_parameter, data)
    # data.close()

    # # read file
    # data = open("data.pkl", "rb")
    # output = pickle.load(data)
    # print(output["Private Key"].sk)
    # data.close()