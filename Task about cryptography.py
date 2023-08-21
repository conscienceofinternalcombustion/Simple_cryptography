import numpy as np
from numpy.linalg import det, inv
from gmpy2 import invert
 
def encode(text, abc):
    return [abc.index(c) for c in text]
 
def decode(data, abc):
    return "".join([abc[v] for v in data])
 
def get_inverse_key(key, M):
    d = int(det(key))
    return np.array(np.round(inv(key) * d) * int(invert(d, M)), dtype=int) % M
 
text = input('текст ')
abc = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
M = len(abc)
key = np.array([
        [1, 2, 0],
        [0, 1, 4],
        [1, 2, 2]
    ])
 
data = encode(text, abc)
keyinv = get_inverse_key(key, M)
r = np.dot(np.array(data).reshape(-1, key.shape[0]), keyinv.T) % M
pt = decode(r.flatten(), abc)
print(pt)
