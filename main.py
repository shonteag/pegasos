#!/usr/bin/env python3
#I know python better, now. Should be prettier. Yay!

import sys, math, time, operator
import vocabGen, vectorGen, pegasos


#---------EMAIL TESTING----------
if "-validate" in sys.argv:
    spam_train_generator = vocabGen.vocabGen()
    spam_train_use_vocab,spam_train_emails = spam_train_generator.build_vocab('data/spam_train.txt', 4000)
    emails_use = spam_train_emails[0:4000]
    emails_validate = spam_train_emails[4000:5000]
    
    spam_train_vecgen = vectorGen.vectorGen()
    spam_train_vectors = spam_train_vecgen.build_feature_arrays(spam_train_use_vocab, spam_train_emails)
    use_vectors = spam_train_vectors[0:4000]
    validate_vectors = spam_train_vectors[4000:5000]
    
    lambda_val = math.pow(2,-5)
    peggy = pegasos.pegasos()
    weights = peggy.pegasos_svm_train(use_vectors,spam_train_emails,lambda_val)
    
    peggy.pegasos_svm_test(weights,validate_vectors,emails_validate)
#---------END EMAIL TESTING--------

#---------TESTING ON NEW DATA------
elif "-test" in sys.argv:
    spam_train_generator = vocabGen.vocabGen()
    train_vocab,train_emails = spam_train_generator.build_vocab('data/spam_train.txt', 5000)
    
    spam_test_generator = vocabGen.vocabGen()
    test_vocab,test_emails = spam_test_generator.build_vocab('data/spam_test.txt', 1000)
    
    spam_train_vecgen = vectorGen.vectorGen()
    train_vectors = spam_train_vecgen.build_feature_arrays(train_vocab, train_emails)
    
    spam_test_vecgen = vectorGen.vectorGen()
    test_vectors = spam_test_vecgen.build_feature_arrays(train_vocab, test_emails)
    
    lambda_val = math.pow(2,-5)
    peggy = pegasos.pegasos()
    weights = peggy.pegasos_svm_train(train_vectors, train_emails,lambda_val)
    
    peggy.pegasos_svm_test(weights,test_vectors,test_emails)