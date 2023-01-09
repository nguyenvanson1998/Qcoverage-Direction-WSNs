import numpy as np

from ableplaceQcover.config import Config
from ableplaceQcover.individual import Individual
from tqdm import tqdm
from time import time
class GA:
    def __init__(self, problem, path):
        self.problem = problem
        self.population = []
        self.convergence = []
        self.path = path

    def init_population(self):
        # random init all
        for _ in range(Config.POPULATION_SIZE):
            self.population.append(Individual(len(self.problem.sensors),self.problem.max_active))

        # heuristic sensor angle adjustment
        for p in self.population[:(Config.POPULATION_SIZE // 2)]:
            self.problem.heuristic_sensor_angle_adjustment(p)
            

    def get_best(self):
        best = self.population[0]
        for p in self.population:
            if p.fitness[0] > best.fitness[0]:
                best = p
            elif p.fitness[0] == best.fitness[0] and p.fitness[1] > best.fitness[1]:
                best = p

        return best

    def reproduction(self):
        offspring = []
        #st = time()
        crossover_rate = np.random.rand(Config.POPULATION_SIZE)

        indx_cross = np.where(crossover_rate < Config.POPULATION_SIZE)[0]

        for idx in indx_cross:
            p1 = self.population[idx]
            p2 = p1
            while p2 == p1:
                p2 = np.random.choice(self.population)
            #st = time() 
            offspring.extend(Individual.crossover(p1, p2, self.problem.get_custom_phi(p1),
                                                      self.problem.get_custom_phi(p2)))
            #print(f'crossover--- time = {time() -st}')
        for p in offspring:
            if np.random.random() < Config.MUTATION_RATE:
                p.mutate(self.problem.get_custom_phi(p))
        
        for p in offspring:
            p.fix_gen()


        return offspring
            

        # for _ in range(Config.POPULATION_SIZE):
        #     # crossover
        #     p1 = np.random.choice(self.population)
        #     p2 = p1
        #     st1 = time()
        #     while p2 == p1:
        #         p2 = np.random.choice(self.population)
        #     print(f'chooosed time = {time() -st1}')
        #     if np.random.random() < Config.CROSSOVER_RATE:
        #         offspring.extend(Individual.crossover(p1, p2, self.problem.get_custom_phi(p1),
        #                                               self.problem.get_custom_phi(p2)))
        #         # mutation
        # print(f'crossover time = {time() -st}')
        # st = time()
        # for p in offspring:
        #     p.fix_gen()
        # print(f'fix time = {time() -st}')
        #     # else:
        #     #     offspring.extend([p1.copy(), p2.copy()])
        # return offspring




    def selection(self, offspring):
        self.population.extend(offspring)
        self.population.sort(key=lambda p: p.fitness, reverse=True)
        self.population = self.population[:Config.POPULATION_SIZE]

    def run(self):
        # init population
        #st = time()
        self.init_population()
        for p in self.population:
            p.fix_gen()
        #print(f'ínit time ={time() - st}')
        # for p in self.population:
        #     print("......",len(np.where(p.active == True)[0]))
        #st = time()
        for p in self.population:
            p.fitness = self.problem.evaluate(p)
        best = self.get_best()
        #print(f'evaluate 0 time ={time() - st}')
        #print(f'Generation 0, best fitness = {best.fitness}')
        #print(f'Generation 0, best fitness = {best.fitness}')
        self.convergence.append(best)

        # evolve
        conve = 0
        for k in tqdm(range(Config.MAX_GENERATION),desc = self.path):
            #st = time()
            offspring = self.reproduction()
            #print(f'reproduct1 time ={time() - st}')
            #st = time()

            for p in offspring:
                # if p.fitness != (0,0):
                    p.fitness = self.problem.evaluate(p)
            #print(f"eval 1 = {time()- st}")
            #st = time()
            self.selection(offspring)
            #print(f'selection 1 = {time() - st}')
            best1 = self.get_best()
            if best1.fitness[0] - best.fitness[0] < - 1e-5:
                conve = conve +1
            elif abs(best1.fitness[0] - best.fitness[0]) <=1e-5:
                if best1.fitness[1] - best.fitness[1] <= 1e-5:
                    conve = conve +1
                else:
                    conve = 0
            else:
                conve = 0
            best = best1
            self.convergence.append(best)
            if conve >= Config.CONVERGING:
                break

            #print(f'Generation {k}, best fitness = {best.fitness}')
            

        return best
