import numpy as np

from nonplaceQcover.config import Config
from nonplaceQcover.individual import Individual
from tqdm import tqdm
class GA:
    def __init__(self, problem, path):
        self.problem = problem
        self.population = []
        self.convergence = []
        self.path = path

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
        for _ in range(Config.POPULATION_SIZE):
            # crossover
            p1 = np.random.choice(self.population)
            p2 = p1
            while p2 == p1:
                p2 = np.random.choice(self.population)
            if np.random.random() < Config.CROSSOVER_RATE:
                offspring.extend(Individual.crossover(p1, p2, self.problem.get_custom_phi(p1),
                                                      self.problem.get_custom_phi(p2)))
            # else:
            #     offspring.extend([p1.copy(), p2.copy()])
        return offspring

    def mute(self):
        # mutation
        for p in self.population:
            if np.random.random() < Config.MUTATION_RATE:
                p.mutate(self.problem.get_custom_phi(p))



    def selection(self, offspring):
        self.population.extend(offspring)
        self.population.sort(key=lambda p: self.problem.evaluate(p), reverse=True)
        self.population = self.population[:Config.POPULATION_SIZE]

    def run(self):
        # init population
        self.init_population()

        for p in self.population:
            p.fitness = self.problem.evaluate(p)
        best = self.get_best()
        #print(f'Generation 0, best fitness = {best.fitness}')
        self.convergence.append(best)

        # evolve
        conve = 0
        for k in tqdm(range(Config.MAX_GENERATION),desc = self.path):

            offspring = self.reproduction()
            self.mute()
            for p in offspring:
                p.fitness = self.problem.evaluate(p)
            self.selection(offspring)
            best1 = self.get_best()
            if best1.fitness[0] - best.fitness[0] < - 1e-5:
                conve = conve +1
            elif abs(best1.fitness[0] - best.fitness[0]) <=1e-5:
                if best1.fitness[1] - best.fitness[1] <= 1e-5:
                    conve = conve +1
                else:
                    conve = 0
            else:
                conve =0
            best = best1
            self.convergence.append(best)
            if conve == Config.CONVERGING:
                break

            #print(f'Generation {k}, best fitness = {best.fitness}')
            

        return best
