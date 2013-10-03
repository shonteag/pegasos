#!/usr/bin/env python3

import sys,operator,time

class vectorGen(object):
    f_vectors = []
    
    
    def __init__(self):
        self.f_vectors = [];
    
    
    #-------------------BUILD FEATURE VECTORS-----------------------
    
    #BEGIN BUILD_FEATURE_ARRAYS
    #vocab_data is vocab dictionary built from specified training data
    #emails_data is list of email hashes
    def build_feature_arrays(self, vocab_data, emails_data):
        if "-silent" not in sys.argv:
            print "Creating feature vectors for " + str(len(emails_data)) + " emails..."
            print " ->Iterating over emails..."
        
        self.f_vectors = [[] for i in range(0,len(emails_data))];
    
        start = time.time(); #run-time computation
        
        #start algorithm
        for index,email_hash in enumerate(emails_data): #number of hashed emails
            
            for key in vocab_data: #loop through all keys to check.
                if email_hash.get(key) != None: #if the word exists in the email hash
                    self.f_vectors[index].append(1)
                else:
                    self.f_vectors[index].append(0)
            
        #end algorithm
        
    
        fin = time.time() - start; #run-time computation
        
        if "-silent" not in sys.argv:
            print " ->Complete. " + str(fin) + " seconds";
            print "   ->" + str(len(self.f_vectors)) + " feature vectors generated of length " + str(len(self.f_vectors[1]));
        
        return self.f_vectors;
    #END BUILD_FEATURE_ARRAYS
        
    
    #-------------------END BUILD FEATURE VECTORS-------------------
    
