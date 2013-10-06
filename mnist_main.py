#!/usr/bin/env python3

import sys,math,itertools
import mnist_vectorGen,mnist_pegasos

#lmbdapower arg
if "-lambdapower" in sys.argv:
    lambda_val = math.pow(2, int(sys.argv[sys.argv.index('-lambdapower') + 1]))
else:
    lambda_val = math.pow(2,-3)
    


vecGen = mnist_vectorGen.mnist_vectorGen()
data = vecGen.setup('data/mnist_train.txt')
#data will be returned as a list of tuples. tuple[0] is the class, tuple[1] is the feature vector
vecGen2 = mnist_vectorGen.mnist_vectorGen()
test_data = vecGen2.setup('data/mnist_test.txt')
    
#peggy is a fickle wench in python, and as such we must use one for each weight vector for the time being
peggy0 = mnist_pegasos.pegasos()
peggy1 = mnist_pegasos.pegasos()
peggy2 = mnist_pegasos.pegasos()
peggy3 = mnist_pegasos.pegasos()
peggy4 = mnist_pegasos.pegasos()
peggy5 = mnist_pegasos.pegasos()
peggy6 = mnist_pegasos.pegasos()
peggy7 = mnist_pegasos.pegasos()
peggy8 = mnist_pegasos.pegasos()
peggy9 = mnist_pegasos.pegasos()
    

def train_weights(data, lambda_val):
    weight0 = peggy0.multiclass_pegasos_train(data, 0, lambda_val)
    weight1 = peggy1.multiclass_pegasos_train(data, 1, lambda_val)
    weight2 = peggy2.multiclass_pegasos_train(data, 2, lambda_val)
    weight3 = peggy3.multiclass_pegasos_train(data, 3, lambda_val)
    weight4 = peggy4.multiclass_pegasos_train(data, 4, lambda_val)
    weight5 = peggy5.multiclass_pegasos_train(data, 5, lambda_val)
    weight6 = peggy6.multiclass_pegasos_train(data, 6, lambda_val)
    weight7 = peggy7.multiclass_pegasos_train(data, 7, lambda_val)
    weight8 = peggy8.multiclass_pegasos_train(data, 8, lambda_val)
    weight9 = peggy9.multiclass_pegasos_train(data, 9, lambda_val)    

    return (weight0,weight1,weight2,weight3,weight4,weight5,weight6,weight7,weight8,weight9)
    
#-------PROBLEM 1a---------------------------------------------
if "-multi" in sys.argv:
    
    mult_weights = train_weights(data, lambda_val)
    
    testPeggy = mnist_pegasos.pegasos()
    miss,error = testPeggy.multiclass_pegasos_test(test_data, mult_weights, (0,1,2,3,4,5,6,7,8,9))
    
#-------CROSS VALIDATION---------------------------------------
elif "-crossvalidate" in sys.argv:
    #would be nice if this implementation could be within the pegasos object itself
    #but I unfortunately don't have the time to make it work
    
    #args
    if "-split" in sys.argv:
        SPLIT_K = int(sys.argv[sys.argv.index('-split') + 1])
    else:
        SPLIT_K = 5
    
    miss = 0
    
    print "Splitting feature vectors into " + str(SPLIT_K) + "..."
    interval = len(data) / SPLIT_K
    split_data = [[] for i in range(0,SPLIT_K)]
    
    start = 0
    for i in range(0,SPLIT_K):
        finish = start + interval
        split_data[i-1] = data[start:finish]
        start = finish  
    print " ->Complete. Split data into " + str(SPLIT_K) + " parts of " + str(len(split_data[0]))
    
    
    new_data = []
    
    for i in range(0,SPLIT_K):
        del new_data[:]
        for x in range(0,SPLIT_K):
            if x != i:
                new_data = list(itertools.chain(new_data,split_data[x]))
            else:
                #skip this part of the training data
                pass
        
        print "TRAINING WEIGHTS ON HOLDOUT " + str(i+1)
        mult_weights = train_weights(new_data, lambda_val)
        
        testPeggy = mnist_pegasos.pegasos()
        miss += testPeggy.multiclass_pegasos_test(split_data[i], mult_weights, (0,1,2,3,4,5,6,7,8,9))
    
    missavg = float(miss) / float(SPLIT_K)
    erroravg = float(float(miss) / float(len(data))) / float(SPLIT_K)
    print "Average misclassification: " + str(missavg)
    print "Average validation error:  " + str(erroravg)


#-------HELP--------------------------------------
else:
    print "Error: No algorithm specified."
    print "Program Arguments"
    print "  -multi              Runs multiclass pegasos on data."
    print "    -lambdapower #    Sets lambda to 2^#. Default # is -3"
    print "  -crossvalidate      Runs cross validation algorithm on split data"
    print "    -lambdapower #    Sets lambda to 2^#. Default # is -3"
    print "    -split #          Sets number to split data. Default # is 5"
    print "Ex: python mnist_main.py -crossvalidate -lambdapower -1 -split 5"
    sys.exit("Please re-run program without -help argument, and specify an algorithm.")
    
    
    