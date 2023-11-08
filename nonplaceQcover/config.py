import numpy as np


class Config:
    # GA parameters
    CROSSOVER_RATE = 0.9
    MUTATION_RATE = 0.1
    MAX_GENERATION = 1000
    POPULATION_SIZE = 100

    # crossover & mutation
    ETA = 6
    DELTA_MAX = 0.05
    CONVERGING = 100
