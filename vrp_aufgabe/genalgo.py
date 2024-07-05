import random
import math
import time
import numpy as np
import evrp

# same as in notebook, here as a default
def crossover(parent1,parent2,instance):
    """
    pick random tour of parent1, pick random tour of parent2 and replace 
    first tour by second tour
    then fix solution, ie, remove customers that appear twice and insert customers that have been lost
    (lost customers are greedily inserted)
    """
    ### exchange single vehicle
    first_car=random.randint(0,len(parent1)-1)
    second_car=random.randint(0,len(parent2)-1)
    child=deepcopy(parent1)  ## need to do deep copy
    child[first_car]=parent2[second_car].copy()
    fix(child,first_car,instance)
    return child

class GenAlgo:
    def __init__(self,instance,params={},crossover=crossover,mutation_methods=[],record=True):
        self.instance=instance
        self.set_params(params)
        self.record=record
        self.best=math.inf
        self.best_animal=None
        self.crossover_method=crossover
        self.mutation_methods=mutation_methods
        self.run()
    
    def set_params(self,params):
        self.population_size=params.get("POP_SIZE",100)
        self.time_budget=params.get("TIME_BUDGET",60)
        self.selection_size=params.get("SELECTION_SIZE",self.population_size//3)
        ks=params.get("KEEP_SIZE",0.05)
        if type(ks) is int:
            self.keep_size=ks
        else:
            self.keep_size=round(self.population_size*ks)
    
    def run(self):
        start=time.time()
        population=self.compute_initial_population()
        self.pop_over_generations=[]
        generation=0
        while time.time()-start<self.time_budget:
            generation+=1
            scores=self.compute_scores(population)
            if self.record:
                self.pop_over_generations.append(deepcopy(population))
            self.update_best(scores,population)
            if self.keep_size>0:
                indices=np.argpartition(scores,self.keep_size)
            child_population=[]
            for i in range(self.population_size-self.keep_size):
                parent1=self.select(population,scores)
                parent2=self.select(population,scores)
                child=self.__crossover(parent1,parent2)
                child=self.__mutate(child)            
                child_population.append(child)
            ## keep best of parent population
            child_population.extend([population[i] for i in indices[:self.keep_size]])
            population=child_population
        if self.record:
            print("time elapsed: {}s".format(round(time.time()-start)))
            print("number of generations: {}".format(len(self.pop_over_generations)))
            #return best_animal,pop_over_generations
        #return best_animal

    def compute_initial_population(self):
        """
        generate tours in random order, then run heuristic to insert charging stations as needed
        """
        population=[]
        for _ in range(self.population_size):
            tour=evrp.rnd_tour(self.instance)
            evrp.fix_range(tour,self.instance)
            population.append(tour)
        return population
            
    def compute_scores(self,population):
        """
        compute objective function for every solution
        """
        scores=[]
        for tour in population:
            score=evrp.soft_objective_function(tour,self.instance)
            scores.append(score)
        return scores
    
    def update_best(self,scores,population):
        """
        keep track of best solution seen so far
        returns current best solution
        """
        for score,animal in zip(scores,population):
            if score<self.best:
                self.best=score
                self.best_animal=animal

    def select(self,population,scores):
        """pick best out of SELECTION_SIZE many random animals"""
        N=len(population)
        choice=random.sample(range(N),self.selection_size)
        best=math.inf
        best_index=None
        for index in choice:
            if scores[index]<best:
                best=scores[index]
                best_index=index
        return population[best_index]


    def __crossover(self,parent1,parent2):
        return self.crossover_method(parent1,parent2,self.instance)

    def __mutate(self,tour):
        # each mutate method: (method,mutation_rate,how_often)
        for mutate_method,mutation_rate,mutation_repetition in self.mutation_methods:
            for _ in range(mutation_repetition):
                if random.random()<=mutation_rate:
                    tour=mutate_method(tour,self.instance)
        return tour

def deepcopy(tour):
    return [vehicle_tour.copy() for vehicle_tour in tour]

def find_missing_and_doubles(child,instance):
    doubles=[]
    counted=[instance.depot]
    for vehicle_tour in child:
        for stop in vehicle_tour:
            if stop in instance.customers:
                if stop in counted and stop!=instance.depot:
                    doubles.append(stop)
                else:
                    counted.append(stop)
    missing=[customer for customer in instance.customers if not customer in counted]
    return missing,doubles

INSERT_TRIES=10
def best_insert_single_missing(tour,missing,instance):
    """
    greedy insertion heuristic for missing customers 
    """
    best=math.inf
    best_pos=None
    for _ in range(INSERT_TRIES):
        vehicle_tour=random.choice(tour)
        pos=random.randint(0,len(vehicle_tour))
        vehicle_tour.insert(pos,missing)
        score=evrp.soft_objective_function([vehicle_tour],instance)
        if score<best:
            best=score
            best_pos=(vehicle_tour,pos)
        del vehicle_tour[pos]
    vehicle_tour,pos=best_pos
    vehicle_tour.insert(pos,missing)

def best_insert_all_missing(tour,missings,instance):
    random.shuffle(missings)
    for m in missings:
        best_insert_single_missing(tour,m,instance)
        
def strip_doubles(child,exchanged_car,doubles):
    for i,vehicle_tour in enumerate(child):
        if i==exchanged_car:
            continue
        stripped=[stop for stop in vehicle_tour if not stop in doubles]
        child[i]=stripped
    
def fix(child,exchanged_car,instance):
    missings,doubles=find_missing_and_doubles(child,instance)
    strip_doubles(child,exchanged_car,doubles)
    best_insert_all_missing(child,missings,instance)
