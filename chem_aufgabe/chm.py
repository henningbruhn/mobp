import random
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


########################### all ####################################
def show_demand(instance):
    fig,axs=plt.subplots(instance.C,1,figsize=(20,2*instance.C),sharex=True,sharey=True)
    axs=iter(axs.flat)
    for c in range(instance.C):
        ax=next(axs)
        demand=[instance.d[c][t] for t in range(instance.T)]
        ax.bar(range(instance.T),demand,color='b',width=0.2,align='edge',alpha=0.8,label="Bedarf "+chems[c]+" #"+str(c))
        ax.set_ylabel("Menge")
        #ax.set_title("Bedarf "+chems[c])
        ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(range(instance.T))
        
    ax.set_xlabel("Zeit")

chems=['Chlor', 'Kohlenstoff', 'Schwefel', 'Kryptonit', 'Wasserstoff', 'Buckyballs']

def colour_pick(quantity):
    if quantity>0:
        return 'g'
    return 'gray'


######################  notebook 1 ##################################
class instance:
    def __init__(self,C=0,T=0,g=0,B=0,h=0,d=None):
        self.C=C
        self.T=T
        self.g=g
        self.B=B
        self.d=d
        self.h=h
        
    def __repr__(self):
        result= "C Anzahl Rohstoffe: ...................{}\n".format(self.C)
        result+="T Anzahl Zeitperioden: ................{}\n".format(self.T)
        result+="B Kapazität des Zwischenlagers: .......{}\n".format(self.B)
        result+="g Kosten einer Fahrt: .................{}\n".format(self.g)
        result+="h Kapazität des Lastwagens: ...........{}\n".format(self.h)
        result+="d Gesamtbedarf über alle Zeitperioden: {}\n".format(np.array(self.d).sum())
        return result
    
    def __str__(self):
        return self.__repr__()

def rnd_instance1(C=10,T=20,seed=None):
    random.seed(seed)
    np.random.seed(seed=seed)
    g=1
    B=100
    h=50
    # Poisson...
    d=np.zeros(shape=(C,T))
    for c  in range(C):
        times=[t for t in np.cumsum(scipy.stats.poisson.rvs(3,size=10)+1)-1 if t < T]
        for t in times:
            d[c,t]=random.randint(5,50)
    return instance(C=C,T=T,g=g,B=B,d=d.tolist(),h=h)

def show_solution1(x,z,p,instance,chemical='all'):
    fig,axs=plt.subplots(2,1,figsize=(20,6),sharex=True)
    axs=axs.flat
    axs=iter(axs)
    ax=next(axs)

    if chemical=='all':
        chemicals=range(instance.C)
        chemical_string="gesamt"
    else:
        chemicals=[chemical]
        chemical_string=chems[chemical]
    demand=[sum([instance.d[c][t] for c in chemicals]) for t in range(instance.T)]
    storage=[sum([p[c][t].x for c in chemicals]) for t in range(instance.T)]
    deliveries=[sum([x[c][t].x for c in chemicals]) for t in range(instance.T)]
    ax.bar(range(instance.T),demand,color='b',width=0.2,align='edge',alpha=0.8,label="Bedarf")
    ax.bar(range(instance.T),deliveries,color='g',width=-0.2,align='edge',alpha=0.8,label="Lieferungen")
    ax.step(range(instance.T),storage,'k',linewidth=4,alpha=0.8,label="Lagerstand",where='post')
    ax.set_ylabel("Menge")
    ax.set_title("Bedarf / Lagerstand / Lieferungen: "+chemical_string)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(range(instance.T))
    
    offset=0.15
    ax=next(axs)
    truck_rides=[[t-1+offset,t-offset] for t in range(instance.T) if z[t].x==1]
    colours=['g']*len(truck_rides)
    if chemical!='all':
        colours=[colour_pick(x[chemical][t].x) for t in range(instance.T) if z[t].x==1]
    for ride,colour in zip(truck_rides,colours):
        ax.plot(ride,[1]*len(ride),colour,linewidth=20)
    ax.set_xlabel("Zeit")
    ax.set_title("Lastwagenfahrten")
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    plt.show()

### sample instance for first notebook  
def get_instance1():
    return rnd_instance1(C=3,T=11,seed=42)

######################  notebook 2 ##################################

class instance2:
    def __init__(self,C=0,T=0,g=0,B=0,h=0,d=None,I=[],r=0,k=0):
        self.C=C
        self.T=T
        self.g=g
        self.B=B
        self.d=d
        self.h=h
        self.I=I
        self.r=r
        self.k=k

    def __repr__(self):
        result= "C Anzahl Rohstoffe: ...................{}\n".format(self.C)
        result+="T Anzahl Zeitperioden: ................{}\n".format(self.T)
        result+="B Kapazität des Zwischenlagers: .......{}\n".format(self.B)
        result+="g Kosten einer Fahrt: .................{}\n".format(self.g)
        result+="h Kapazität der Lastwagen: ............{}\n".format(self.h)
        result+="d Gesamtbedarf über alle Zeitperioden: {}\n".format(np.array(self.d).sum())
        result+="k Anzahl der Lastwagen: ...............{}\n".format(self.k)
        result+="r Fahrtzeit der Lastwagen: ............{}\n".format(self.r)
        result+="I gefährliche Rohstoffpaare: ..........{}\n".format(self.I)
        return result
    
    def __str__(self):
        return self.__repr__()
        
def rnd_instance2(C=10,T=20,k=5,r=3,incompatability=0.1,seed=None):
    random.seed(seed)
    np.random.seed(seed=seed)
    g=1
    B=100
    h=10
    # Poisson...
    d=np.zeros(shape=(C,T))
    for c  in range(C):
        times=[t for t in np.cumsum(scipy.stats.poisson.rvs(3,size=10)+1)-1 if t < T]
        for t in times:
            d[c,t]=random.randint(5,50)
    I=[]
    for c in range(C):
        for cc in range(c+1,C):
            if random.random()<=incompatability:
                I.append((c,cc))
    return instance2(C=C,T=T,g=g,B=B,d=d,h=h,I=I,k=k,r=r)

def show_solution2(x,z,p,instance,chemical='all'):
    fig,axs=plt.subplots(2,1,figsize=(20,6),sharex=True)
    axs=axs.flat
    axs=iter(axs)
    ax=next(axs)

    if chemical=='all':
        chemicals=range(instance.C)
        chemical_string="gesamt"
    else:
        chemicals=[chemical]
        chemical_string=chems[chemical]
    demand=[sum([instance.d[c][t] for c in chemicals]) for t in range(instance.T)]
    storage=[sum([p[c][t].x for c in chemicals]) for t in range(instance.T)]
    deliveries=[sum([x[c][t][v].x for c in chemicals for v in range(instance.k)]) for t in range(instance.T)]
    ax.bar(range(instance.T),demand,color='b',width=0.2,align='edge',alpha=0.8,label="Bedarf")
    ax.bar(range(instance.T),deliveries,color='g',width=-0.2,align='edge',alpha=0.8,label="Lieferungen")
    ax.step(range(instance.T),storage,'k',linewidth=4,alpha=0.8,label="Lagerstand",where='post')
    ax.set_ylabel("Menge")
    ax.set_title("Bedarf / Lagerstand / Lieferungen: "+chemical_string)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(range(-instance.r,instance.T))
    
    offset=0.1
    ax=next(axs)
    for truck in range(instance.k):
        truck_rides=[[t-instance.r+offset,t-offset] for t in range(instance.T) if z[t][truck].x==1]
        colours=['g']*len(truck_rides)
        if chemical!='all':
            colours=[colour_pick(x[chemical][t][truck].x) for t in range(instance.T) if z[t][truck].x==1]
        for ride,colour in zip(truck_rides,colours):
            ax.plot(ride,[truck+1]*len(ride),colour,linewidth=10)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Lastwagen")
    ax.set_title("Lastwagenfahrten")
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

### sample instance for second notebook  
def get_instance2():
    return rnd_instance2(C=4,T=12,k=7,r=3,incompatability=0.1,seed=42)

######################  notebook 3 ##################################

class instance3:
    def __init__(self,C=0,T=0,g=0,b=None,h=0,d=None,I=[],r=0,k=0,f=0):
        self.C=C
        self.T=T
        self.g=g
        self.b=b
        self.d=d
        self.h=h
        self.I=I
        self.r=r
        self.k=k
        self.f=f
        self.S=len(b)

    def __repr__(self):
        result= "C Anzahl Rohstoffe: ...................{}\n".format(self.C)
        result+="T Anzahl Zeitperioden: ................{}\n".format(self.T)
        result+="g Kosten einer Fahrt: .................{}\n".format(self.g)
        result+="h Kapazität der Lastwagen: ............{}\n".format(self.h)
        result+="d Gesamtbedarf über alle Zeitperioden: {}\n".format(np.array(self.d).sum())
        result+="k Anzahl der Lastwagen: ...............{}\n".format(self.k)
        result+="r Fahrtzeit der Lastwagen: ............{}\n".format(self.r)
        result+="I gefährliche Rohstoffpaare: ..........{}\n".format(self.I)
        result+="S Anzahl Zwischenlager: ...............{}\n".format(self.S)
        result+="b Kapazitäten Zwischenlager: ..........{}\n".format(self.b)
        result+="f Strafkosten späte Bereitstellung: ...{}\n".format(self.f)
        return result
    
    def __str__(self):
        return self.__repr__()
        
        
def rnd_instance3(C=10,T=20,k=5,r=3,S=4,h=18,f=100,how_many=1,seed=None):
    random.seed(seed)
    np.random.seed(seed=seed)
    g=1
    # Poisson...
    d=np.zeros(shape=(C,T))
    for c  in range(C):
        times=[t for t in np.cumsum(scipy.stats.poisson.rvs(3,size=10)+1)-1 if t < T]
        for t in times:
            d[c,t]=random.randint(5,50)
    I=[]
    while len(I)<how_many:
        c=random.randint(0,C-2)
        cc=random.randint(c+1,C-1)
        if not (c,cc) in I:
            I.append((c,cc))
    b=[random.randint(20,40) for _ in range(S)]
    return instance3(C=C,T=T,g=g,b=b,d=d,h=h,I=I,k=k,r=r,f=f)

def show_solution3(x,z,p,w,instance,chemical='all'):
    fig,axs=plt.subplots(3,1,figsize=(20,9),sharex=True)
    axs=axs.flat
    axs=iter(axs)
    ax=next(axs)

    if chemical=='all':
        chemicals=range(instance.C)
        chemical_string="gesamt"
    else:
        chemicals=[chemical]
        chemical_string=chems[chemical]
    demand=[sum([instance.d[c][t] for c in chemicals]) for t in range(instance.T)]
    storage=[sum([p[c][t][s].x for c in chemicals for s in range(instance.S)]) for t in range(instance.T)]
    deliveries=[sum([x[c][t][v].x for c in chemicals for v in range(instance.k)]) for t in range(instance.T)]
    tardy=[sum([w[c][t].x for c in chemicals]) for t in range(instance.T)]
    ax.bar(range(instance.T),demand,color='b',width=0.2,align='edge',alpha=0.8,label="Bedarf")
    ax.bar(range(instance.T),deliveries,color='g',width=-0.2,align='edge',alpha=0.8,label="Lieferungen")
    ax.step(range(instance.T),storage,'k',linewidth=4,alpha=0.8,label="Lagerstand",where='post')
    ax.step(range(instance.T),tardy,'r--',linewidth=4,alpha=0.8,label="verspätet",where='post')
    ax.set_ylabel("Menge")
    ax.set_title("Bedarf / Lagerstand / Lieferungen: "+chemical_string)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(range(-instance.r,instance.T))
    
    ax=next(axs)
    offset=0.1
    for truck in range(instance.k):
        truck_rides=[[t-instance.r+offset,t-offset] for t in range(instance.T) if z[t][truck].x==1]
        colours=['g']*len(truck_rides)
        if chemical!='all':
            colours=[colour_pick(x[chemical][t][truck].x) for t in range(instance.T) if z[t][truck].x==1]
        for ride,colour in zip(truck_rides,colours):
            ax.plot(ride,[truck+1]*len(ride),colour,linewidth=10)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Lastwagen")
    ax.set_title("Lastwagenfahrten")
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax=next(axs)
    for s in range(instance.S):
        storage_indicators=[]
        current=[]
        for t in range(instance.T):
            if sum([p[c][t][s].x for c in chemicals])>0:
                current.append(t)
            else:
                if len(current)>0:
                    current.append(t)
                    storage_indicators.append(current)
                    current=[]
        if len(current)>0:
            storage_indicators.append(current)
        colour='k'
        for current in storage_indicators:
            ax.plot(current,[s+1]*len(current),colour,linewidth=10)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Lager")
    ax.set_title("Lagernutzung")
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks(range(1,instance.S+1))
    plt.show()

### sample instance for third notebook  
def get_instance3():
    return rnd_instance3(C=4,T=11,k=6,h=25,seed=3456,how_many=2)



######################  notebook 4 ################################## 



    
# Sample Instances notebook 4
def get_instances4():
    n = 10
    seed = 0
    random.seed(seed)
    np.random.seed(seed=seed)
    inner_seeds = [random.randrange(100000) for i in range(n)]
    instances = [rnd_instance3(C=4,T=11,k=6,h=15,f=0.1,seed=s,how_many=2) for s in inner_seeds ]
    # Append extra b
    for inst in instances:
        inst.b.append(50)
    return instances
    