import random
import sys
import time
from gmpy2 import mpz, is_prime,random_state,mpz_urandomb


rand = random_state(random.randrange(sys.maxsize))

def running_time(f, c = 1):
    # function cost time
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        clockTime = time2 - time1
        if c == 0:

            return clockTime
        else:
            print(f"{f.__name__} function took{clockTime * 1000000.0: 0.3f}us")
            return ret
    return wrap

def generate_prime(bits):
    """Generate an integer of b bits that is prime
       using the gmpy2 library"""

    while True:
        possible = mpz(2) ** (bits - 1) + mpz_urandomb(rand, bits - 1)
        if is_prime(possible):
            return possible

if __name__ == "__main__":
    # testing
    bits = 1024
    p = generate_prime(bits)
    print(p)
    print(p.bit_length())

    t_generate_prime = running_time(generate_prime, c = 0)
    clock_time_all = 0
    for i in range(10):
        clockTime = t_generate_prime(bits)
        clock_time_all = clock_time_all + clockTime
    clock_time_avg = clock_time_all / 10.0
    print(f"Average time for generating a prime of length {bits} bits: {clock_time_avg * 1000: 0.3f}ms")