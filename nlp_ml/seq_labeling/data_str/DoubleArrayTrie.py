import numpy as np


class DoubleArrayTrie:
    def __init__(self):
        self.base_array = np.array([])
        self.check_array = np.array([])

    def get_base(self, n):
        return self.base_array[n]

    def get_check(self, m):
        return self.check_array[m]

    def move_child(self, n, ch):
        k = ord(ch)
        m = self.get_base(n) + k
        if not self.get_check(m) == n:
            return -1
        return m
