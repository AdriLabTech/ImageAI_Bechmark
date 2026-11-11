"""splitmix64 y Box-Muller (docs/dataset.md §5.2, docs/architecture.md §7.1). Solo herramientas, nunca motores."""
import math

MASK = (1 << 64) - 1
SEED = 20261110


class SplitMix64:
    def __init__(self, seed):
        self.x = seed & MASK

    def next(self):
        self.x = (self.x + 0x9E3779B97F4A7C15) & MASK
        z = self.x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def uniform(self):
        """u ∈ [0, 1) con 53 bits."""
        return (self.next() >> 11) * 2.0 ** -53

    def normal_pair(self):
        """Box-Muller: consume dos salidas (a, b) → (z0, z1). u1 ∈ (0, 1] evita log(0)."""
        u1 = ((self.next() >> 11) + 1) * 2.0 ** -53
        u2 = (self.next() >> 11) * 2.0 ** -53
        r = math.sqrt(-2.0 * math.log(u1))
        return r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)

    def shuffle(self, items):
        """Fisher-Yates in situ: j = siguiente() mod (i+1)."""
        for i in range(len(items) - 1, 0, -1):
            j = self.next() % (i + 1)
            items[i], items[j] = items[j], items[i]
