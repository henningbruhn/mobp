### rnlt2.py --- helper code for RENAULT exercise in MOBP ###
### includes also greedy code ###


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import math
import random


class Vehicle:
    """represents all data on a car that needs to be scheduled
    ident: id number of car
    colour: the colour the car needs to be painted with
    options: list of options that need to be installed
    """
    def __init__(self,ident,colour,options):
        self.ident=ident
        self.colour=colour
        self.options=options
        
    def __repr__(self):
        return str(self.ident)
    
    def __str__(self):
        return "id: {}, Farbe: {}, Optionen: {}".format(self.ident,self.colour,self.options)

### reading and parsing instance data ###

def get_weights(opt_objectives):
    weights={'paint':0, 'high':0, 'low':0}
    w=[1000000,1000,1]
    for i,row in opt_objectives.iterrows():
        key=re.match(r"([a-z]+)_",row['objective name']).group(1)
        weights[key]=w[row['rank']-1]
    return weights
    
def read_objectives(filepath):
    opt_objectives=pd.read_csv(filepath,sep=';')
    opt_objectives=opt_objectives.drop(columns="Unnamed: 2")
    return get_weights(opt_objectives)
    
def parse_paint_batch_limit(filepath):
    paint_batch_limit_csv=pd.read_csv(filepath,sep=';')
    paint_batch_limit=paint_batch_limit_csv.loc[0,'limitation']
    return paint_batch_limit    

def read_ratios(filepath):
    ratios_table=pd.read_csv(filepath,sep=';')
    ratios_table['p']=ratios_table["Ratio"].str.extract(r"(\d*)/").astype('int')
    ratios_table['q']=ratios_table["Ratio"].str.extract(r"/(\d*)").astype('int')
    ratios_table=ratios_table.drop(columns=['Unnamed: 3'])
    ratios={}
    for i,row in ratios_table.iterrows():
        ratios[row['Ident']]=(row['p'],row['q'])
    return ratios

def process_vehicles(vehicles):
    previous_day_vehicles=[]
    current_day_vehicles=[]
    for i,row in vehicles.iterrows():
        options=[option for option in row[4:-1].index if row[option]==1]
        car=Vehicle(row["Ident"],row['Paint Color'],options)
        if row['already scheduled']:
            previous_day_vehicles.append(car)
        else:
            current_day_vehicles.append(car)
    return previous_day_vehicles,current_day_vehicles

def read_vehicles(filepath):
    vehicles=pd.read_csv(filepath,sep=';')
    previous_day=vehicles['Date'].iloc[0]
    vehicles["already scheduled"]=vehicles['Date']==previous_day
    return vehicles

def get_renault_schedule(vehicles):
    """
    Die Reihung des Renault-Algorithmus
    """
    renault_schedule=list(vehicles.loc[vehicles['already scheduled']==False,'Ident'])
    return renault_schedule

def read_in_all_instances(path,silent=False):
    data_dict={}
    for root, dirs, files in os.walk(path):
        instance_dict={}
        rest,first=os.path.split(root)
        data_dict[first]=instance_dict
        for filename in files:
            filepath=os.path.join(root,filename)
            if filename=="optimization_objectives.txt":
                opt_obj=read_objectives(filepath)
                instance_dict["weights"]=opt_obj            
            if filename=="paint_batch_limit.txt":
                paint_batch_limit=parse_paint_batch_limit(filepath)
                instance_dict['paint_batch_limit']=paint_batch_limit
            if filename=="ratios.txt":
                ratios=read_ratios(filepath)
                instance_dict["ratios"]=ratios            
            if filename=="vehicles.txt":
                vehicles=read_vehicles(filepath)
                previous,current=process_vehicles(vehicles)
                #instance_dict["previous_day"]=previous   # we don't use this
                instance_dict["current_day"]=current
                instance_dict["renault_schedule"]=current.copy()
    delete=[]
    for key in data_dict.keys():
        if data_dict[key]=={}:
            delete.append(key)
        data_dict[key]['name']=key
    for key in delete:
        del data_dict[key]
    if not silent:
        print("Folgende Instanzen wurden eingelesen: ")
        for key in data_dict.keys():
            print("  "+key)
    return data_dict

from operator import itemgetter
def prio_string(instance):
    prio=[what for what,weight in sorted(instance['weights'].items(),key=itemgetter(1),reverse=True)]
    return "{} >> {} >> {}".format(prio[0],prio[1],prio[2])
    
### sanity check ###

def check_for_completeness(schedule,instance,chatty=True):
    for car in instance['current_day']:
        if not car in schedule:
            if chatty:
                print("Fahrzeug {} nicht in schedule".format(car))
            return False
    if chatty:
        print("Reihung vollständig!")
    return True


### objective function, helper code for visualisation ###
    
def compute_colour_changes_list(schedule,instance):
    """
    liefert Liste mit Einträge 0,1 pro Fahrzeug zurück. 1 bedeutet vor Fahrzeug muss gereinigt werden. Berücksichtigt paint_batch_limit
    """
    current_colour=schedule[0].colour
    batch_count=1
    colour_changes=[0]*len(schedule)
    for i,car in enumerate(schedule[1:]):
        batch_count+=1
        if car.colour!=current_colour:
            current_colour=car.colour
            colour_changes[i+1]=1
            batch_count=1
        if batch_count==instance['paint_batch_limit']+1:
            colour_changes[i+1]=1
            batch_count=1
    return colour_changes

def compute_colour_changes(schedule,instance):
    return sum(compute_colour_changes_list(schedule,instance))

def compute_option_usage(schedule,instance):
    """wir ignorieren die Produktion vom Vortag"""
    ratios=instance['ratios']
    usage={}
    for option in ratios.keys():
        p,q=ratios[option]
        options_indicator=[(option in car.options) for car in schedule]
        sliding_sum=np.convolve(options_indicator,np.ones(q))
        usage[option]=sliding_sum
    return usage

def compute_option_violations(schedule,instance):
    usage=compute_option_usage(schedule,instance)
    ratios=instance['ratios']
    violations={}
    for option in ratios.keys():
        p,q=ratios[option]
        violations[option]=sum(np.maximum(usage[option]-p,0))
    return violations
    
def compute_scores(schedule,instance):
    score=compute_option_violations(schedule,instance)
    score['colour']=compute_colour_changes(schedule,instance)
    return score

def compute_objective_by_score(scores,instance):
    weights=instance['weights']
    objective=0
    for key in scores.keys():
        score=scores[key]
        if key=="colour":
            objective+=weights['paint']*score
        elif key[0]=='L':
            objective+=weights['low']*score
        elif key[0]=='H':
            objective+=weights['high']*score
    return objective    

def compute_objective(schedule,instance):
    scores=compute_scores(schedule,instance)
    return compute_objective_by_score(scores,instance)
    
### visualisation ###

def plot_colour_changes(schedule,instance,ax=None):
    if ax is None:
        fig,ax=plt.subplots(1,1,figsize=(20,2))
    change_list=compute_colour_changes_list(schedule,instance)
    N=len(change_list)
    ax.set_yticks([])
    ax.set_title('colour changes: {}'.format(sum(change_list)))
    ax.bar(range(N),change_list,color='b',width=1.5)
    
def plot_options_and_colours(schedule,instance):
    ratios=instance['ratios']

    plot_num=len(ratios.keys())+1
    fig,axs=plt.subplots(plot_num,1,figsize=(20,plot_num*2),gridspec_kw={"hspace":0.5},subplot_kw={})
    axs=axs.flat
    ### colour changes first
    plot_colour_changes(schedule,instance,ax=axs[0])
    
    ### now options
    usage=compute_option_usage(schedule,instance)
    violations=compute_option_violations(schedule,instance)
    for option,ax in zip(ratios.keys(),axs[1:]):
        p,q=ratios[option]
        
        N=len(usage[option])
        ax.step(range(N),usage[option],'b',where='mid')
        ax.plot(range(N),[p]*N,'r',alpha=0.8)
        ax.set_title(option+" {}/{} * violations: {}".format(p,q,violations[option]))
    plt.show()    
    
### greedy ###

import copy
class Best_Tracker:
    """convenience class to keep track of best solution"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.best_cost=np.inf
        self.best=None
    
    def update(self,solution,cost):
        if cost<self.best_cost:
            self.best_cost=cost
            self.best=copy.copy(solution) ## we could simply write solution.copy() -- but this also works with strings
            return True
        return False


def greedy(instance):
    current_day=instance['current_day']
    remaining_cars=list(np.random.permutation(current_day))
    first_car=remaining_cars.pop() ## erstes Auto, das gereiht werden muss
    partial_schedule=[first_car] ## wir starten mit einer Reihung von einem Auto
    while len(remaining_cars)>0: ## solange noch Autos gereiht werden müssen...
        next_car=pick_next_greedily(partial_schedule,remaining_cars,instance)
        partial_schedule.append(next_car)
        remaining_cars.remove(next_car)
    return partial_schedule

def penalty_increase(next_car,partial_schedule,instance):
    """Um wie viel erhöhen sich die Kosten, wenn next_car als nächstes gereiht wird?"""
    partial_schedule.append(next_car)
    ## Um zu sehen, wie die Kosten sich verändern, müssen wie nicht die ganze Reihung 
    ## betrachten -- das Endstück reicht. Wir nehmen hier die letzten 30 Autos. 
    ## Da zu Beginn die Reihung auch weniger als 30 Autos enthalten kann, müssen 
    ## wir ein wenig aufpassen, wie lang das Endstück sein soll
    window_length=min(len(partial_schedule),30)
    cost_increase=compute_objective(partial_schedule[-window_length:],instance)
    del partial_schedule[-1]  ## entferne next_car wieder
    return cost_increase

def pick_next_greedily(partial_schedule,remaining_cars,instance,tries=50):
    """finde nächstes Auto für Reihung, das Kosten nur gering erhöht"""
    best_car=None
    best_penalty_increase=np.inf
    for _ in range(tries):
        car=random.choice(remaining_cars)
        ## um wie viel erhöhen sich die Kosten, wenn car als nächstes ausgesucht wird?
        increase=penalty_increase(car,partial_schedule,instance)
        ## wenn besser als bisher bestes, nimm car stattdessen
        if increase<best_penalty_increase:
            best_penalty_increase=increase
            best_car=car
    return best_car


