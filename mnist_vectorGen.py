#!/usr/bin/env python3
#Don't need vocab list, since we already know all featurs are 0-255

import sys,math,operator,time
from copy import deepcopy

class mnist_vectorGen(object):
    temp_vector = []
    vectors = []

    
    def __init__(self):
        self.temp_vector = []
        self.vectors = []


    def setup(self, f):
        print "Setting up MNIST hashes and feature vectors..."
        print " ->Data file: " + str(f)
        start = time.time()
        raw_data = file(f, 'r')
        
        for index,line in enumerate(raw_data):
            
            thisline = line.split(',')
            
            del self.temp_vector[:]
            self.temp_vector = [0 for i in range(0,len(thisline))]
            
            for index,val in enumerate(thisline):
                if index > 0:
                    vector_val = ((2 * int(val)) / 255) - 1
                    self.temp_vector[index] = vector_val
                else:
                    #this is the designator, dont manipulate it
                    pass
            
            
            #now add this new vector to the list of vectors
            self.vectors.append((int(thisline[0]),deepcopy(self.temp_vector)))
        
        fin = time.time() - start
        print " ->Generated " + str(len(self.vectors)) + " vectors."
        print " ->Complete. " + str(fin) + " seconds."
        
        return self.vectors
        
    
    

