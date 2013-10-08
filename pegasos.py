#!/usr/bin/env python3

import sys,operator,time,math,decimal

class pegasos(object):
    #public vars
    weights = []
    objective = []
    MAX_ITER = 20
    
    def __init__(self):
        #initialize public vars and structs
        weights = []
        MAX_ITER = 20
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
        objective_val = ((lambda_val/2) * math.pow(self.magnitude(weights),2)) + (float(1)/float(m)) * part2
        return objective_val
    
    def hinge_loss(self, m, part2):
        return (float(1)/float(m)) * part2
    
    
    
    
    def pegasos_svm_train(self, vectors, email_data, lambda_val, sv): #lambda_val is the regularization constant
        print "Training PEGASOS algorithm..."
        
        weights = [0 for i in range(0,len(vectors))]
        u = [0 for i in range(0,len(vectors))]
        
        
        iteration = 1
        t = 0
        eta = 0
        total_hinge_loss = 0
        while True:
            
            propclass = 0
            impropclass = 0
            
            #for eval function
            part2 = 0
            
            for index,vector in enumerate(vectors):
                t += 1
                label = 1 if (email_data[index].get('0') == None) else -1
                eta = float(1) / (t * lambda_val)
                
                u = [0 for i in range(0,len(vectors))]
                
                if (label * self.dot(weights,vector)) < 1:
                    #print "< 1: " + str(label * self.dot(weights,vector))
                    #modify weight vector accordingly
                    u = self.add(self.scalar_mult((1-(eta * lambda_val)),weights),self.scalar_mult((eta*label),vector))
                    impropclass += 1
                else:
                    #print ">= 1: " + str(label * self.dot(weights,vector))
                    #modify weight vector accordingly
                    u = self.scalar_mult((1-(eta * lambda_val)),weights)
                    propclass += 1
                
                minimum = min(1,(1/(math.sqrt(lambda_val)))/self.magnitude(u))
                temp = self.scalar_mult(minimum,u)
                weights = temp[:]
                
                part2 += max(0,1-(label * self.dot(weights,vector)))
                
                del u[:]
                
            #evaluate objective function
            objective_val = self.eval(weights, len(vectors), lambda_val, part2)
            hinge_loss = self.hinge_loss(len(vectors), part2)
            print "   ->Iteration " + str(iteration)
            print "       Objective Val = " + str(objective_val)
            print "       Hinge Loss = " + str(hinge_loss)
        
            total_hinge_loss += hinge_loss
        
            if iteration >= self.MAX_ITER:
                break
        
            iteration += 1

        print " ->Average hinge loss = " + str((total_hinge_loss / float(self.MAX_ITER)))
        
        #calculate support vectors
        if sv:
            print " ->Calculating number of support vectors..."
            num_sv = 0
            for index,vector in enumerate(vectors):
                a = decimal.Decimal(self.dot(weights,vector))
                a_round = round(a,2)
                if (a_round == 1) or (a_round == -1):
                    num_sv += 1
                
            print "  ->Complete. Found " + str(num_sv) + " support vectors."
        
        return weights
    
    
    def pegasos_svm_test(self, weights, feature_vectors, email_data):
        print "Running test data..."
        print " ->Running on " + str(len(feature_vectors));
        
        start = time.time();
        
        num_errors = 0
        
        for index,vector in enumerate(feature_vectors):
            label = 1 if (email_data[index].get('0') == None) else -1;
            result = self.checker(vector,weights)
            if result == label:
                #correctly classified
                continue;
            else:
                num_errors += 1;
        
        fin = time.time() - start;
        
        misclassified = (float(num_errors) / float(len(feature_vectors)))
        print " ->Complete. " + str(fin) + " seconds."
        print "   ->Test Error: " + str(misclassified) + "; Misclassified " + str(num_errors) + " out of " + str(len(feature_vectors)) + " vectors.";
        return misclassified;
    
    
    
    
    
    
