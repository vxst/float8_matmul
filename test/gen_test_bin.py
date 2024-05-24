import numpy as np
from ml_dtypes import float8_e5m2


# Generate a binary file with 10 group of random float8_e5m2 numbers
# 4 items in each group
def gen_vector():
    return np.random.rand(10, 4).astype(float8_e5m2)


def gen_add_test_bin():
    with open('test_add.bin', 'wb') as f:
        a = gen_vector()
        b = gen_vector()
        c = a.astype(np.float64) + b.astype(np.float64)
        c = c.astype(float8_e5m2)
        f.write(a.tobytes())
        f.write(b.tobytes())
        f.write(c.tobytes())


def gen_fma_test_bin():
    with open('test_fma.bin', 'wb') as f:
        a = gen_vector()
        b = gen_vector()
        c = gen_vector()
        d = (a.astype(np.float64) * b.astype(np.float64)).astype(float8_e5m2)
        d = (d.astype(np.float64) + c.astype(np.float64)).astype(float8_e5m2)
        f.write(a.tobytes())
        f.write(b.tobytes())
        f.write(c.tobytes())
        f.write(d.tobytes())


if __name__ == '__main__':
    gen_add_test_bin()
    gen_fma_test_bin()
    print(float8_e5m2(128.0).tobytes().hex())