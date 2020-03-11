
import random

import numpy as np
import matplotlib.pyplot as plt


# PARAMETERS                                            #

popSize = 100
chromLength = 300
iteration_max = 100
crossover_rate = 0.7
mutation_rate = 0.001

fitness = np.empty([popSize])
costVector = np.empty([chromLength])

# Load network                                          #
def loadNetwork():

    fname = "network.txt"
    input = np.loadtxt(fname)
    for i in range(0,chromLength):

        costVector[i]=input[i][2]




# FITNESS EVALUATION                                    #

def evaluateFitness(chromosome,best):
    costFullyConnectedNetwork=30098.059999999983
    fitness_total=0.0
    fitness_average=0.0

    for i in range(0,popSize):
        fitness[i]=0

    for i in range(0,popSize):
        cost=0
        for j in range(0,chromLength):
            if chromosome[i,j]==1:
                cost=cost+costVector[j]
        fitness[i]=1-(cost/costFullyConnectedNetwork)
        fitness_total=fitness_total+fitness[i]
    fitness_average=fitness_total/popSize

    for i in range(0,popSize):
        if fitness[i]>=best:
            best = fitness[i]

    return best, fitness_average






# PERFORMANCE GRAPH                                     #

def plotChart(best,avg):
    plt.plot(best,label='best')
    plt.plot(avg,label='average')
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.legend()
    plt.xlim(1,iteration_max-1)
    plt.ylim(0, 1)
    plt.show()



#SELECTION OF THE BEST CHROMOSOME FROM THE TWO BY COMPARING FITNESS VALUES
def tournamentSelection1(chromosomePopulation,popSize,best):
        #initialise i
        i=0
        Value1=np.random.randint(0,popSize)
        Value2=np.random.randint(0,popSize)
        chro1=chromosomePopulation[Value1]
        chro2=chromosomePopulation[Value2]
        #STORING THE FITNESS VALUES OF THE RESPECTIVE CHROMOSOMES
        best1=fitness[Value1]
        best2=fitness[Value2]
        #COMPARING THE FITNESS VALUES TO FINOUT THE BEST ONE

        if best1 >best2 :
            i=Value1
        else:
            i=Value2
        return i


def tournamentSelection2(chromosomePopulation,popSize,best):
         #initialise i
        i=0
        Value1=np.random.randint(0,popSize)
        Value2=np.random.randint(0,popSize)
        chro1=chromosomePopulation[Value1]
        chro2=chromosomePopulation[Value2]
        #STORING THE FITNESS VALUES OF THE RESPECTIVE CHROMOSOMES
        best1=fitness[Value1]
        best2=fitness[Value2]
        #COMPARING THE FITNESS VALUES TO FINOUT THE BEST ONE
        if best1 >best2 :
            i=Value1
        else:
            i=Value2
        return i




#MUTATION
#G1 is first gene, g2 IS SECOND GENE
def mutation(G1,G2,chromLength):
  #RANDOM VALUE BETWEEN O AND 1
  r=np.random.randint(0,1)
  if r < mutation_rate :
      mutationPoint=np.random.randint(0,chromLength)
    #If we have 0, change to 1 and if we have 1 change to 0
      if G1[mutationPoint]==0 :
          G1[mutationPoint]=1
      else :
          G1[mutationPoint]=0

      if  G2[mutationPoint]==0 :
          G2[mutationPoint]=1
      else :
          G2[mutationPoint]=0

      return G1,G2

  else :
    return G1,G2




#CROSSOVER
def crossover(G1,G2,chromLength):

        #generate random number between o and 1 , so that given mutation rate will be good
        r=np.random.randint(0,1)
        if r < crossover_rate :
            Point1=np.random.randint(0,chromLength)
            Point2=np.random.randint(0,chromLength)

            if Point1>Point2:
                #replacing the crossover points
                Point1,Point2=Point2,Point1
            else:
                Point1=Point1
                Point2=Point2

            G1[Point1:Point2],G2[Point1:Point2]=G2[Point1:Point2],G1[Point1:Point2]
            return G1,G2
        else:
            return G1,G2





#########################################################
# MAIN                                                  #
#########################################################
if __name__ == '__main__':
    best=0.0
    average=0.0
    iteration=0
    bestM = np.empty([iteration_max],dtype=np.float32)
    averageM = np.empty([iteration_max],dtype=np.float32)
    print("GENETIC ALGORITHM APPLIED TO OVERLAY NETWORK OPTIMIZATION")

    chromosome_size=(popSize,chromLength)
    chromosomePopulation=np.random.randint(2,size=chromosome_size)
    #ind=0
    loadNetwork()
    best,fitness_average=evaluateFitness(chromosomePopulation,best)
    tmp_pop=[]
    while (iteration<=iteration_max-1):
        #... to be implemented
        i=0
        tmp_pop=[]
        best=0.0
        average=0.0
        while i<50:
            G1=tournamentSelection1(chromosomePopulation,popSize,best)
            G2=tournamentSelection2(chromosomePopulation,popSize,best)
            G1=chromosomePopulation[G1]
            G2=chromosomePopulation[G2]
            G1,G2=crossover(G1,G2,chromLength)

            G1,G2=mutation(G1,G2,chromLength)
            tmp_pop.append(G1)
            tmp_pop.append(G2)
            print("I:",i)
            i=i+1
        chromosomePopulation=(np.array(tmp_pop))
        best,fitness_average=evaluateFitness(chromosomePopulation,best)
        bestM[iteration] = best
        averageM[iteration] = fitness_average
        print("best fitness: ", best)
        print("average fitness: ", fitness_average)

        print("Iteration:",iteration)
        iteration=iteration+1
    print("#########################")

    print("best fitness: ", best)
    print("average fitness: ", fitness_average)
    print(bestM,averageM)
    plotChart(bestM,averageM)



#%%
