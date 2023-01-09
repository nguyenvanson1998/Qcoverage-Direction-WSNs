import numpy as np


class Config:
    # GA parameters
    CONVERGING = 80
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1
    MAX_GENERATION = 500
    POPULATION_SIZE = 100

    # crossover & mutation
    ETA = 6
    DELTA_MAX = 0.05
