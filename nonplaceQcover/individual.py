import numpy as np

from nonplaceQcover.config import Config


class Individual:
    def __init__(self, dim: int):
        self.dim = dim
        self.active = np.random.choice([True, False], self.dim)
        self.alpha = np.random.uniform(0.0, 2 * np.pi, self.dim)
        self.fitness = 0

    def copy(self):
        p = Individual(self.dim)
        p.dim = self.dim
        p.active = np.copy(self.active)
        p.alpha = np.copy(self.alpha)
        return p

    @staticmethod
    def crossover(p1, p2):
        o1 = p1.copy()
        o2 = p2.copy()

        # active crossover - uniform
        r = np.random.random(p1.dim)
        o1.active = np.where(r < 0.5, p1.active, p2.active)
        o2.active = np.where(r < 0.5, p2.active, p1.active)

        # alpha crossover - sbx
        u = np.random.random(size=p1.dim)
        beta = np.where(u < 0.5, (2 * u) ** (1 / (Config.ETA + 1)),
                        (0.5 / u) ** (1 / (Config.ETA + 1)))

        o1.alpha = ((1 + beta) * p1.alpha + (1 - beta) * p2.alpha) / 2
        o1.alpha = np.clip(o1.alpha, 0, 2 * np.pi)
        o2.alpha = ((1 - beta) * p1.alpha + (1 + beta) * p2.alpha) / 2
        o2.alpha = np.clip(o2.alpha, 0, 2 * np.pi)

        return o1, o2

    def mutate(self):
        # active mutate - bit flip
        mutated_idx = np.random.randint(0, self.dim)
        self.active[mutated_idx] = not self.active[mutated_idx]

        # alpha mutate - polynomial
        u = np.random.random(size=self.dim)
        alpha = self.alpha / (2 * np.pi)
        sigma = np.where(u < 0.5, np.power(2 * u + (1 - 2 * u) * np.power(1 - alpha, Config.ETA + 1),
                                         1 / (Config.ETA + 1)) - 1,
                         1 - np.power(2 * (1 - u) + 2 * (u - 0.5) * np.power(alpha, Config.ETA + 1),
                                    1 / (Config.ETA + 1)))

        self.alpha += Config.DELTA_MAX * sigma * (2 * np.pi)
        self.alpha = np.clip(self.alpha, 0, 2 * np.pi)
