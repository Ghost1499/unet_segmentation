import numpy as np


def main():
    rng1 = np.random.default_rng(0)
    print(rng1.random(5))
    print(rng1.random(5))
    rng2 = np.random.default_rng(0)
    print(rng2.random(5))


if __name__ == "__main__":
    main()
