import numpy as np
import matplotlib.pyplot as plt
### Wir benutzen das fivethirtyeight style, damit die plots ein bisschen hübscher aussehen
### könnte auch weggelassen werden
plt.style.use('fivethirtyeight')
import math
import pandas as pd

############################ read in instances ###############################

class Instance:
    def __init__(self,name,attributes,nodes,customers,stations,demands,depot):
        self.name=name
        self.depot=depot
        self.nodes=nodes
        self.demands=demands
        self.customers=customers
        self.stations=stations
        self.attributes=attributes
        self.set_attributes()

    def set_attributes(self):
        self.vehicles=self.attributes["VEHICLES"]
        self.num_customers=len(self.customers)
        self.num_stations=len(self.stations)
        self.capacity=self.attributes['CAPACITY']
        self.energy_capacity=self.attributes['ENERGY_CAPACITY']
        self.energy_consumption=self.attributes['ENERGY_CONSUMPTION']
        self.optimum=self.attributes['OPTIMAL_VALUE']

    def dist(self,p,q):
        # dist always Euclidean
        px,py=self.nodes[p]
        qx,qy=self.nodes[q]
        return math.sqrt((px-qx)**2+(py-qy)**2)

    def truncate_to(self,field,length):
        length=max(length,10)
        result=str(field)
        if len(result)<=length:
            return result
        return result[:length-5]+",..."+result[-1]

    def __repr__(self):
        repr="instance\n"
        repr+=" .name=              {}\n".format(self.name)
        repr+=" .num_customers=     {}\n".format(self.num_customers)
        repr+=" .vehicles=          {}\n".format(self.vehicles)
        repr+=" .num_stations=      {}\n".format(self.num_stations)
        repr+=" .capacity=          {}\n".format(self.capacity)
        repr+=" .energy_capacity=   {}\n".format(self.energy_capacity)
        repr+=" .energy_consumption={}\n".format(self.energy_consumption)
        repr+=" .depot=             {}\n".format(self.depot)
        repr+=" .customers=         {}\n".format(self.truncate_to(self.customers,30))
        repr+=" .demands=           {}\n".format(self.truncate_to(self.demands,30))
        repr+=" .stations=          {}\n".format(self.truncate_to(self.stations,30))
        repr+=" .nodes=             {}\n".format(self.truncate_to(self.nodes,30))
        repr+=" .optimum=           {}\n".format(self.optimum)
        return repr
        

        

class Instance_Reader:
    def __init__(self,filename):
        self.filename=filename
        self.attributes={}
        self.depot=None
        self.nodes={}
        self.demands={}
        self.customers=[]
        self.stations=[]
        self.demands={}
        self.process_line=None
        self.run()

    def process_attribute(self,line):
        splits=line.split(':')
        value="".join(splits[1:]).strip()
        try:
            value=int(value)
        except ValueError:
            try:
                value=float(value)
            except ValueError:
                pass
        self.attributes[splits[0]]=value

    def toggle_section(self,line):
        if line=="NODE_COORD_SECTION":
            self.process_line=self.process_node_line
        elif line=="DEMAND_SECTION":
            self.process_line=self.process_demand_line
        elif line=="DEPOT_SECTION":
            self.process_line=self.process_depot_line
        elif line=="STATIONS_COORD_SECTION":
            self.process_line=self.process_station_line
        elif line=="EOF":
            self.process_line=self.process_EOF
        else:
            return False
        return True

    def process_node_line(self,line):
        splits=line.split(' ')
        self.nodes[int(splits[0])]=(int(splits[1]),int(splits[2]))

    def process_demand_line(self,line):
        splits=line.split(' ')
        customer=int(splits[0])
        self.customers.append(customer)
        self.demands[customer]=int(splits[1])
    
    def process_depot_line(self,line):
        value=int(line)
        if value>=0:
            self.depot=value
        
    def process_station_line(self,line):        
        self.stations.append(int(line))

    def process_EOF(self,line):
        pass

    def run(self):
        with open(self.filename,'r') as f:
            for line in f:
                line=line.strip()
                if ':' in line:
                    self.process_attribute(line)
                elif not self.toggle_section(line):
                    self.process_line(line)
        self.setup_instance()

    def setup_instance(self):
        if not self.depot in self.stations:
            self.stations.append(self.depot) # depot counts as charging station
        base_filename=os.path.split(self.filename)[1]
        self.instance=Instance(base_filename[:-5],self.attributes,self.nodes,self.customers,self.stations,self.demands,self.depot)
   
import os

def read_in_all_instances(path):
    instances={}
    for filename in os.listdir(path):
        filepath=os.path.join(path,filename)
        if filename.endswith(".evrp"):
            inst=Instance_Reader(filepath).instance
            instances[filename[:-5]]=inst
    return instances

################ plot #####################################################

def get_scaler(instance):
    vals=np.array(list(instance.demands.values()))
    dmin,dmax=min(vals),max(vals)
    
    def scale(demand):
        return 50+(demand-dmin)/(dmax-dmin)*100
    
    return scale

def plot_vehicle_tour(vehicle_tour,instance,ax,vehicle_number=None):
    for i,_ in enumerate(vehicle_tour[1:]):
        start=instance.nodes[vehicle_tour[i]]
        stop=instance.nodes[vehicle_tour[i+1]]
        ax.plot([start[0],stop[0]],[start[1],stop[1]],'gray',linewidth=3,zorder=-1)
    if vehicle_number is not None:
        ax.set_title('Fahrzeug #{}'.format(vehicle_number))
            
def plot_inst(instance,ax):
    depot=instance.nodes[instance.depot]
    ax.scatter([depot[0]],[depot[1]],s=150,marker='s',label='Depot',color='b')
    size_scaler=get_scaler(instance)
    for customer in instance.customers:
        if customer==instance.depot:
            continue
        ax.scatter([instance.nodes[customer][0]],[instance.nodes[customer][1]],color='r',marker='x',s=size_scaler(instance.demands[customer]))

    for station in instance.stations:
        if station==instance.depot:
            continue
        ax.scatter([instance.nodes[station][0]],[instance.nodes[station][1]],color='b',marker='^',s=100)

    vals=np.array(list(instance.nodes.values()))
    xmax,xmin=max(vals[:,0]),min(vals[:,0])
    ymax,ymin=max(vals[:,1]),min(vals[:,1])
    xmarge=0.05*(xmax-xmin)
    ymarge=0.05*(ymax-ymin)

    for label,marker,color in [('E-Tanke','^','b'),('Kunde','x','r')]:
        ax.scatter([],[],marker=marker,label=label,color=color)
        pass
        
    ax.set_xlim(xmin-xmarge,xmax+xmarge)
    ax.set_ylim(ymin-ymarge,ymax+3*ymarge)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left")
    
def show_attributes(instance,ax):
    index=['DIMENSION','VEHICLES','STATIONS','CAPACITY','ENERGY_CAPACITY','ENERGY_CONSUMPTION','OPTIMAL_VALUE']
    df=pd.DataFrame([[round(instance.attributes[attr],1)] for attr in index if attr in instance.attributes.keys()],index=index,columns=['Instanz'])
    df=df.rename(index={'DIMENSION':'Kunden','VEHICLES':'Fahrzeuge','OPTIMAL_VALUE':'Optimum', 'STATIONS':'Ladestationen', \
        'CAPACITY':'Kapazität','ENERGY_CAPACITY':'Batteriekapazität','ENERGY_CONSUMPTION':'Energieverbrauch/km'})
    font_size=12
    ax.axis('off')
    mpl_table = ax.table(cellText = df.values, rowLabels = df.index, loc='center right',colLabels=df.columns,colWidths=[0.3])
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)    
    
def show(instance,tour=[None],num_cols=3):
    num_rows=(len(tour)+1+num_cols-1)//num_cols
    fig=plt.figure(figsize=(20,20/num_cols*num_rows))
    if tour[0] is not None:
        if validate(tour,instance,quiet=False):
            tour_len=tour_length(tour,instance)
            fig.suptitle("Legale Tour der Länge {}".format(round(tour_len,1)), fontsize=16)
        tour=ensure_depot(tour,instance)
    for i,vehicle_tour in enumerate(tour):
        ax=plt.subplot(num_rows,num_cols,i+1)
        plot_inst(instance,ax)
        if vehicle_tour is not None:
            plot_vehicle_tour(vehicle_tour,instance,ax,vehicle_number=i+1)
    ax=plt.subplot(num_rows,num_cols,len(tour)+1)
    show_attributes(instance,ax)
    #plt.tight_layout()
    plt.show()   

####################### loads, charge, length ##################################

def ensure_depot(tour,instance):
    new_tour=[]
    for vehicle_tour in tour:
        new_vehicle_tour=vehicle_tour.copy()
        if len(new_vehicle_tour)==0 or new_vehicle_tour[0]!=instance.depot:
            new_vehicle_tour.insert(0,instance.depot)
        if new_vehicle_tour[-1]!=instance.depot:
            new_vehicle_tour.append(instance.depot)
        new_tour.append(new_vehicle_tour)
    return new_tour

def compute_charge_lvls(tour,instance):
    tour=ensure_depot(tour,instance)
    energy_consumption_factor=instance.attributes['ENERGY_CONSUMPTION']
    max_charge=instance.attributes['ENERGY_CAPACITY']
    charge_lvls_per_vehicle=[]
    for vehicle_tour in tour:
        charge=max_charge  ## vehicle starts fully charged at depot
        charge_lvls=[charge]
        charge_lvls_per_vehicle.append(charge_lvls)
        for i,stop in enumerate(vehicle_tour[1:]):
            consumption=instance.dist(vehicle_tour[i],stop)*energy_consumption_factor
            charge=charge-consumption
            charge_lvls.append(charge)
            if stop in instance.stations:  ## stop is charging station
                charge=max_charge          ## vehicle charges
    return charge_lvls_per_vehicle

def compute_loads(tour,instance):
    loads=[]
    for vehicle_tour in tour:
        load=0
        for stop in vehicle_tour:
            load+=instance.demands.get(stop,0)
        loads.append(load)
    return loads
    
def tour_lengths(tour,instance):
    tour=ensure_depot(tour,instance)
    lengths=[]
    for vehicle_tour in tour:
        length=0
        for i,_ in enumerate(vehicle_tour[1:]):
            edge_length=instance.dist(vehicle_tour[i],vehicle_tour[i+1])
            length+=edge_length
        lengths.append(length)
    return lengths
        
def tour_length(tour,instance):
    return sum(tour_lengths(tour,instance))
    

############# penalties, soft constraint objective ######################### 

def sum_under_charges(tour,instance):
    charge_lvls=compute_charge_lvls(tour,instance)
    flat_charges=flatten(charge_lvls)
    return -sum([charge for charge in flat_charges if charge<0])

WEIGHT=1E6

def soft_objective_function(tour,instance,weight=WEIGHT):
    loads=compute_loads(tour,instance)
    max_load=instance.attributes['CAPACITY']
    load_penalty=sum([max(load-max_load,0) for load in loads])*weight
    charge_penalty=sum_under_charges(tour,instance)*weight
    return load_penalty+charge_penalty+tour_length(tour,instance)
    
########################## stats / validation #################################

def flatten(list_of_lists):
    flat_list = [item for sublist in list_of_lists for item in sublist]
    return flat_list

def check_for_single_customer(customer,tour):
    for vehicle_tour in tour:
        if customer in vehicle_tour:
            return True
    return False

def check_for_customers(tour,instance,quiet=True):
    for customer in instance.customers:
        if customer==instance.depot:
            continue
        if not check_for_single_customer(customer,tour):
            if not quiet:
                print("Kunde {} wird nicht angefahren!".format(customer))
            return False
    return True
        
def check_for_capacity(tour,instance,quiet=True):
    loads=compute_loads(tour,instance)
    if max(loads)>instance.attributes['CAPACITY']:
        if not quiet:
            print("Ladung {} übersteigt maximale Ladung von {}!".format(max(loads),instance.attributes['CAPACITY']))
        return False
    return True

def check_for_range(tour,instance,quiet=True):
    charge_lvls=compute_charge_lvls(tour,instance)
    if min(flatten(charge_lvls))<0:
        if not quiet:
            print("Reichweite überschritten!")
        return False
    return True

def check_for_num_vehicles(tour,instance,quiet=True):
    if len(tour)>instance.attributes['VEHICLES']:
        if not quiet:
            print("Es werden {} Fahrzeuge eingesetzt, es sind jedoch nur {} Fahrzeuge vorhanden!".format(len(tour),instance.attributes['VEHICLES']))
        return False
    return True
    
def validate(tour,instance,quiet=True):
    num_v=check_for_num_vehicles(tour,instance,quiet=quiet)
    cust=check_for_customers(tour,instance,quiet=quiet)
    rng=check_for_range(tour,instance,quiet=quiet)
    cap=check_for_capacity(tour,instance,quiet=quiet)
    return num_v and cust and rng and cap 

def count_stations(tour,instance):
    tour=ensure_depot(tour,instance)
    counts=[]
    for vehicle_tour in tour:
        station_count=0
        for stop in vehicle_tour[1:-1]:
            if stop in instance.stations:
                station_count+=1
        counts.append(station_count)
    return counts

def vehicle_stats(tour,instance):
    loads=compute_loads(tour,instance)
    charge_lvls=compute_charge_lvls(tour,instance)
    min_charges=[round(min(charges),1) for charges in charge_lvls]
    lengths=[round(length,1) for length in tour_lengths(tour,instance)]
    station_counts=count_stations(tour,instance)
    data=np.array([loads,min_charges,station_counts,lengths]).T
    columns=['Auslastung','Ladungs-Min','#Ladehalts','Streckenlänge']
    index=range(1,len(tour)+1)
    df=pd.DataFrame(data,columns=columns,index=index)
    df.index.name='Fahrzeug'
    return df



################# rnd tour generation ##################################

import random

def tour_vector_to_list(tour_vector):
    vehicle_tour=[]
    tour=[vehicle_tour]
    for stop in tour_vector:
        if stop<0:
            vehicle_tour=[]
            tour.append(vehicle_tour)
        else:
            vehicle_tour.append(stop)
    return tour

def rnd_tour_vector(instance):
    tour_vector=instance.customers.copy()
    num_vehicles=instance.attributes['VEHICLES']
    for i in range(num_vehicles-1):
        tour_vector.append(-(i+1))
    random.shuffle(tour_vector)
    return tour_vector

def insert_rnd_stations(tour_vector,num_stations,instance):
    for _ in range(num_stations):
        pos=random.randint(1,len(tour_vector))
        station=random.choice(instance.stations)
        tour_vector.insert(pos,station)

def rnd_tour(instance,num_stations_insert=0):
    tour_vector=rnd_tour_vector(instance)
    insert_rnd_stations(tour_vector,num_stations_insert,instance)
    return tour_vector_to_list(tour_vector)


############### fix range heuristic ##########################

def insert_station(vehicle_tour,instance,tries=10):
    best=sum_under_charges([vehicle_tour],instance)
    if best==0:
        return 0
    best_pos=-1
    best_station=-1
    for _ in range(tries):
        station=random.choice(instance.stations)
        pos=random.randint(0,len(vehicle_tour))
        vehicle_tour.insert(pos,station)
        under_charge=sum_under_charges([vehicle_tour],instance)
        if under_charge<best:
            best=under_charge
            best_pos=pos
            best_station=station
        del vehicle_tour[pos]
    if best_pos<0:
        return best
    vehicle_tour.insert(best_pos,best_station)
    return best

def insert_stations(vehicle_tour,instance,max_stations=3,tries=10):
    under_charge=np.inf
    for _ in range(max_stations):
        under_charge=insert_station(vehicle_tour,instance,tries=tries)
        if under_charge==0:
            return

def fix_range(tour,instance):
    for vehicle_tour in tour:
        insert_stations(vehicle_tour,instance,max_stations=100,tries=30)
        
        
################### gen algo analytics #########################################

def get_edges(tour):
    edges=[]
    for vehicle_tour in tour:
        for i,_ in enumerate(vehicle_tour[:-1]):
            edges.append((vehicle_tour[i],vehicle_tour[i+1]))
    return edges
        
def similarity(tour1,tour2):
    edges1=get_edges(tour1)
    edges2=get_edges(tour2)
    return sum([1 for edge in edges1 if edge in edges2])/len(edges1)

def diversity(population,num_comparisons=100):
    common=0
    for _ in range(num_comparisons):
        tour1=random.choice(population)
        tour2=random.choice(population)
        common+=similarity(tour1,tour2)
    return 1-common/num_comparisons

def compute_scores_over_gens(pops_over_gens,instance):
    S=[]
    for pop in pops_over_gens:
        S.append([soft_objective_function(p,instance) for p in pop])
    return S

def compute_min_scores(scores_over_gens):
    return [min(scores) for scores in scores_over_gens]

def show_analytics(pops_over_gens,instance):
    FSIZE=12
    scores_over_gens=compute_scores_over_gens(pops_over_gens,instance)
    min_scores=compute_min_scores(scores_over_gens)
    small_values=[score for score in min_scores if score<WEIGHT]
    if len(small_values)>0:
        ymax=max(small_values)
    else:
        ymax=max(min_scores)*1.05
#    fig,axs=plt.subplots(1,4,figsize=(22,5))
    fig,axs=plt.subplots(2,2,figsize=(12,12),sharex=True)
    axs=axs.flat
    axs[0].plot(range(len(min_scores)),min_scores)
    axs[0].set_title("Min Kosten in Generation",fontsize=FSIZE)
    axs[0].set_ylim(0,ymax)
#    axs[0].set_xlabel("Generation",fontsize=FSIZE)
    
    axs[1].plot(range(len(min_scores)),[min(min_scores[:n+1]) for n in range(len(min_scores))])
    axs[1].set_title("Kosten bester Lösung nach x Generationen",fontsize=FSIZE)
    axs[1].set_ylim(0,ymax)
#    axs[1].set_xlabel("Generation",fontsize=FSIZE)
    min_gen=np.argmin(min_scores)
    axs[1].scatter([min_gen],[min_scores[min_gen]],color='r',s=150,alpha=0.8,label="Beste in Gen #{}\nmit Kosten {}".format(min_gen,round(min_scores[min_gen])))
    axs[1].legend()


    N=len(pops_over_gens[0])
    valid_rate=[sum([1 for score in scores if score<WEIGHT])/N*100 for scores in scores_over_gens]
    axs[2].plot(range(len(valid_rate)),valid_rate)
    axs[2].set_title("Anteil (%) legaler Lösungen",fontsize=FSIZE)
    axs[2].set_xlabel("Generation",fontsize=FSIZE)
    axs[2].set_ylim(0,100)

    divs=[diversity(pop) for pop in pops_over_gens]
    axs[3].plot(range(len(divs)),divs)
    axs[3].set_title("Diversität pro Generation",fontsize=FSIZE)
    axs[3].set_xlabel("Generation",fontsize=FSIZE)
    axs[3].set_ylim(0,1)
    
    plt.show()

######################## infeasible sample tours for instance E-n33-k4 #####################

def sample1():
    return [[3, 13, 12, 7, 8, 9, 10, 5],
 [],
 [32, 2, 14, 25, 24, 21, 22, 37, 23, 20, 19, 16, 15, 31],
 [4, 6, 33, 11, 18, 26, 27, 28, 17, 29, 30]]

def sample2():
    return [[5, 9, 7, 6, 20, 22, 21, 23, 24, 25, 17, 34, 18, 4],
 [31, 32, 2, 33, 13, 3],
 [8, 10, 11, 19, 26, 34, 28, 29, 30],
 [38, 12, 14, 16, 27, 15]]

def sample3():
    return [[30, 29, 17, 34, 28, 27, 18, 16],
 [14, 20, 19, 22, 23, 25, 24, 21, 37, 26, 11, 5],
 [4, 6, 7, 8, 10, 9, 33, 31, 36],
 [3, 13, 12, 2, 15, 32]]