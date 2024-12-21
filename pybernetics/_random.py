import time
from ._Typing import RealNumber

class _MersenneTwister:
    period = 2 ** 32 - 1
    def __init__(self, seed):
        # Initialize the state vector with 624 elements
        self.state = [0] * 624
        self.index = 0

        # Initialize the state with the seed
        self.state[0] = seed & 0xffffffff
        for i in range(1, 624):
            self.state[i] = (0x6c078965 * (self.state[i - 1] ^ (self.state[i - 1] >> 30)) + i) & 0xffffffff
        
    def _twist(self):
        for i in range(624):
            y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff)  # Most significant bit
            self.state[i] = self.state[(i + 397) % 624] ^ (y >> 1)
            if y % 2 != 0:
                self.state[i] ^= 0x9908b0df  # A constant

        self.index = 0  # Reset the index after twisting

    def _generate_number(self):
        if self.index == 0:
            self._twist()

        y = self.state[self.index]
        self.index += 1
        
        # Tempering the output to improve randomness
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)

        return y & 0xffffffff  # Return a 32-bit integer

    def next(self):
        return self._generate_number()

mt_seed = int(time.time()) # Truncation
mt = _MersenneTwister(seed = mt_seed)

def random(min: RealNumber = 0.0, max: RealNumber = 1.0) -> float:
    return (mt.next() / mt.period) * (max - min) + min

def _testrun():
    while True:
        minn = float(input("Min: "))
        maxx = float(input("Max: "))
        value = random(min=minn, max=maxx)
        print(f"Result: {value}")
