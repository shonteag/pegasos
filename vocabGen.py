#!/usr/bin/env python3

import sys,operator,time

class vocabGen(object):
    #-------------------VOCAB LIST GENERATION-------------------
    vocab = {}
    temp = {}
    emails = []

    #BEGIN FUNCTION
    def __init__(self):
        self.vocab = {}
        self.temp = {}
        self.emails = []
    
    def parse_hashes(self, i, number): #adds temp_dict to vocab.
        #hash all emails individually for later looping.
        self.emails.append(self.temp.copy());
        
        if i <= number: #number = number of emails used to build vocab list
            #remove 0 and 1 from the temp
            if self.temp.get('0') != None: self.temp.pop('0');
            else: self.temp.pop('1');
            
            for word in self.temp:
                try:
                    self.vocab[word] += 1;
                except KeyError:
                    self.vocab[word] = 1;
        
        self.temp.clear();
        
    #END FUNCTION
    
    #BEGIN FUNCTION
    def write_list_file(self):
        list_file = open('data/list_file.txt', 'w');
        
        print " ->Working on writing list file....";
        for key in self.vocab:
            list_file.write(key + " : " + str(self.vocab[key]) + '\n');
        
        f = open('data/hash.txt','w');
        for email in self.emails:
            f.write(str(email) + '\n');
        f.close();
        
        print " ->Complete.";
    #END FUNCTION
    
    #BEGIN FUNCTION
    def drop_low_liers(self):
        print " ->Dumping low-lier occurence words";
        for key in self.vocab.keys(): #iterate over the vocab list (hashmap)
            if self.vocab[key] < 30: #if word appears in less than 30 distinct emails
                del self.vocab[key];
        print " ->Complete.";
    #END FUNCTION
    
    #BEGIN BUILD_VOCAB
    #file is the file from which to pull raw data (_train.txt or test.txt)
    #number is the number of emails used to build vocab list (4000 or 5000)
    def build_vocab(self, file, number):
        print "Building vocab list using " + str(number) + " emails:";
        start_time = time.time();
        raw_data = open(file, 'r'); #training set of 4000 emails
    
        del self.emails[:];
        self.vocab.clear();
    
        holder = 1; #temp variable to make sure we dont parse empty hashtables
        print " ->Iterating over training emails...";
        i=0;
        for line in raw_data:
            #split the line into full words with the delimeter " " (space)
            thisline = line.split();
            
            for word in thisline:
                
                if self.temp.get(word) == None:
                    self.temp[word] = 1;
            i += 1;
            self.parse_hashes(i, number);
        
        run = time.time() - start_time;
        print " ->Complete. " + str(run) + " seconds";
        #drop all entries with value less than 30.
        self.drop_low_liers();
        print "   ->" + str(len(self.vocab)) + " words in vocab list.";
       
        raw_data.close();
        return self.vocab,self.emails;
    #END BUILD_VOCAB
    
    #-------------------END VOCAB LIST GENERATION-------------------