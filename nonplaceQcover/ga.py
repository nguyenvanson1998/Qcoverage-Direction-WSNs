import numpy as np

from nonplaceQcover.config import Config
from nonplaceQcover.individual import Individual
from time import time

class GA:
    def __init__(self, problem):
        self.problem = problem
        self.population = []
        self.convergence = []

    def init_population(self):
        # random init all
        for _ in range(Config.POPULATION_SIZE):
            self.population.append(Individual(len(self.problem.sensors)))

        # heuristic sensor angle adjustment
        for p in self.population[:(Config.POPULATION_SIZE // 2)]:
            self.problem.heuristic_sensor_angle_adjustment(p)

    def get_best(self):
        best = self.population[0]
        for p in self.population:
            if p.fitness[0] > best.fitness[0] or p.fitness[1] > best.fitness[1]:
                best = p
        return best

    def reproduction(self):
        offspring = []
        while len(offspring) < Config.POPULATION_SIZE:
            # crossover
            p1 = np.random.choice(self.population)
            p2 = p1
            while p2 == p1:
                p2 = np.random.choice(self.population)
            if np.random.random() < Config.CROSSOVER_RATE:
                offspring.extend(Individual.crossover(p1, p2, self.problem.get_custom_phi(p1),
                                                      self.problem.get_custom_phi(p2)))
            else:
                offspring.extend([p1.copy(), p2.copy()])

            # mutation
            for p in offspring:
                if np.random.random() < Config.MUTATION_RATE:
                    p.mutate(self.problem.get_custom_phi(p))

        return offspring

    def selection(self, offspring):
        self.population.extend(offspring)
        self.population.sort(key=lambda p: self.problem.evaluate(p), reverse=True)
        self.population = self.population[:Config.POPULATION_SIZE]

    def run(self):
        # init population
        st = time()
        self.init_population()
        print(f'ínit time ={time() - st}')
        st = time()
        for p in self.population:
            p.fitness = self.problem.evaluate(p)
        best = self.get_best()
        print(f'evaluate 0 time ={time() - st}')
        print(f'Generation 0, best fitness = {best.fitness}')
        self.convergence.append(best)

        # evolve
        for k in range(Config.MAX_GENERATION):
            st = time()
            offspring = self.reproduction()
            print(f'repro 0 time ={time() - st}')
            for p in offspring:
                p.fitness = self.problem.evaluate(p)
            self.selection(offspring)
            best = self.get_best()
            print(f'Generation {k}, best fitness = {best.fitness}')
            self.convergence.append(best)

        return self.get_best()
