# declare a function that finds the prime factors of a number

def prime_factors(n : int): -> list
    """Find the prime factors of a number
    :param n: number to factorize
    :returns list of prime factors
    """
    # initialize a list to hold the prime factors
    factors = []
    for i in range(2, n+1):
        # if i is a factor of n
        if n % i == 0:
            # add i to the list of factors
            factors.append(i)
            # divide n by i
            n = n / i
            # if n is 1, we have found all the prime factors
            if n == 1:
                break