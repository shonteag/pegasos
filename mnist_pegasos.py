#!/usr/bin/env python3
#pegasos learning algorithm. now cleaner, more versatile, and tastier than before!

import sys,operator,time,math
from copy import deepcopy

class pegasos(object):
    #public vars
    weights = []
    objective = []
    MAX_ITER = 5
    
    def __init__(self):
        #initialize public vars and structs
        weights = []
        MAX_ITER = 5
        objective = [float(0) for i in range(0,MAX_ITER)]

    
    def dot(self, vector1, vector2):
        return sum(val1*val2 for val1,val2 in zip(vector1,vector2))
    
    def magnitude(self, vector):
        tot = float(0)
        for val in vector:
            tot += float(math.pow(val,2))
        mag = float(math.sqrt(tot))
        return .0000001 if (tot == 0.0) else mag #cheating to avoid 0 divisor
    
    def add(self, vector1, vector2):
        vector3 = []
        for val1,val2 in zip(vector1,vector2):
            vector3.append(val1 + val2)
        return vector3
    
    def scalar_mult(self, scalar, vector):
        for index,val in enumerate(vector):
            vector[index] *= scalar
        return vector
    
    
    
    
    def checker(self, vector, weights):
        #check if weight vector properly classifies vector
        result = 1 if (self.dot(weights,vector) > 0) else -1
        return result
    
    def eval(self, weights, m, lambda_val, part2):
        objective_val = ((lambda_val/2) * math.pow(self.magnitude(weights),2)) + part2
        return objective_val
    
    
    
    def pegasos_svm_train(self, data, lambda_val): #lambda_val is the regularization constant
        print "Training PEGASOS algorithm..."
        
        t1 = data[0]
        t = t1[1]
        weights = [0 for i in range(0,len(t))]
        u = [0 for i in range(0,len(t))]
        
        
        iteration = 1
        t = 0
        eta = 0
        while True:
            
            propclass = 0
            impropclass = 0
            
            #for eval function
            part2 = 0
            
            for index,tup in enumerate(data):
                y = tup[0]
                vector = tup[1]
                
                t += 1
                label = 1 if (y == 1) else -1
                eta = float(1) / (t * lambda_val)
                
                u = [0 for i in range(0,len(data))]
                 
                if (label * self.dot(weights,vector)) < 1:
                    u = self.add(self.scalar_mult((1-(eta * lambda_val)),weights),self.scalar_mult((eta*label),vector))
                    impropclass += 1
                else:
                    u = self.scalar_mult((1-(eta * lambda_val)),weights)
                    propclass += 1
                
                minimum = min(1,(1/(math.sqrt(lambda_val)))/self.magnitude(u))
                temp = self.scalar_mult(minimum,u)
                weights = temp[:]
                
                part2 += max(0,1-(label * self.dot(weights,vector)))
                
                del u[:]
                
            #evaluate objective function
            objective_val = self.eval(weights, len(data), lambda_val, part2)
            print "   ->Iteration " + str(iteration) +". Objective Value = " + str(objective_val)
        
            if iteration >= self.MAX_ITER:
                break
        
            iteration += 1
        
        return weights
    
    
    def pegasos_svm_test(self, weights, data):
       
        start = time.time();
        
        num_errors = 0
        
        for index,tup in enumerate(data):
            y = tup[0]
            vector = tup[1]
            
            label = 1 if (y == '1') else -1;
            result = self.checker(vector,weights)
            if result == label:
                #correctly classified
                continue;
            else:
                num_errors += 1;
        
        fin = time.time() - start;
        
        misclassified = (float(num_errors) / float(len(feature_vectors)))
        return misclassified;
    
    
    
    def multiclass_pegasos_train(self, data, classifier, lambda_val):
        print "Training PEGASOS weight vector on classifiers: " + str(classifier)
        print "-------------------------------------------------------------------"
        
        new_data = [() for i in range(0,len(data))]
        
        for index,tup in enumerate(data):
            if int(tup[0]) != int(classifier):
                new_tup = (-1, tup[1])
            else:
                new_tup = (1, tup[1])
            
            new_data[index] = deepcopy(new_tup)
           
        return self.pegasos_svm_train(new_data, lambda_val)
        
    
    
    
    def multiclass_pegasos_test(self, test_data, mult_weights, classifiers):
        print "Testing on test data..."
        
        miss = 0
        
        #for all data points
        for index,tup in enumerate(test_data):
            actual_y = tup[0]
            vector = tup[1]
            
            holder = [0 for i in range(0,len(mult_weights))]
            
            #test each datapoint against each weight vector
            for ind,weights in enumerate(mult_weights):
                holder[ind] = self.dot(weights,vector)
            
            #check each result
            max_index,max_val = max(enumerate(holder), key=operator.itemgetter(1))
            #print str(holder)
            if int(classifiers[max_index]) != int(actual_y):
                miss += 1
                #print str(classifiers[max_index]) + " : " + str(actual_y)
        
        print " ->Complete. " + str(miss) + " misses. Classification error: " + str(float(miss) / float(len(test_data)))
        
        
        return miss

    
    
    