#!/usr/bin/env python3
#I know python better, now. Should be prettier. Yay!

import sys, math, time, operator
import vocabGen, vectorGen, pegasos

#lmbdapower arg
if "-l" in sys.argv:
    lambda_pow = int(sys.argv[sys.argv.index('-l') + 1])
else:
    lambda_pow = -5

lambda_val = math.pow(2, lambda_pow)

#support vector args
sv = True if "-sv" in sys.argv else False

#---------EMAIL TESTING----------
if "-validate" in sys.argv:
    print "VALIDATING ON LABMDA = 2^" + str(lambda_pow)
    
    spam_train_generator = vocabGen.vocabGen()
    spam_train_use_vocab,spam_train_emails = spam_train_generator.build_vocab('data/spam_train.txt', 4000)
    emails_use = spam_train_emails[0:4000]
    emails_validate = spam_train_emails[4000:5000]
    
    spam_train_vecgen = vectorGen.vectorGen()
    spam_train_vectors = spam_train_vecgen.build_feature_arrays(spam_train_use_vocab, spam_train_emails)
    use_vectors = spam_train_vectors[0:4000]
    validate_vectors = spam_train_vectors[4000:5000]
    
    #validation error
    peggy = pegasos.pegasos()
    weights = peggy.pegasos_svm_train(use_vectors,spam_train_emails,lambda_val, sv)
    
    peggy.pegasos_svm_test(weights,validate_vectors,emails_validate)
#---------END EMAIL TESTING--------

#---------TESTING ON NEW DATA------
elif "-test" in sys.argv:
    print "TESTING ON LABMDA = 2^" + str(lambda_pow)    

    spam_train_generator = vocabGen.vocabGen()
    train_vocab,train_emails = spam_train_generator.build_vocab('data/spam_train.txt', 5000)
    
    spam_test_generator = vocabGen.vocabGen()
    test_vocab,test_emails = spam_test_generator.build_vocab('data/spam_test.txt', 1000)
    
    spam_train_vecgen = vectorGen.vectorGen()
    train_vectors = spam_train_vecgen.build_feature_arrays(train_vocab, train_emails)
    
    spam_test_vecgen = vectorGen.vectorGen()
    test_vectors = spam_test_vecgen.build_feature_arrays(train_vocab, test_emails)
    
    peggy = pegasos.pegasos()
    weights = peggy.pegasos_svm_train(train_vectors, train_emails,lambda_val, sv)
    
    #test error
    print "TEST ERROR:"
    peggy.pegasos_svm_test(weights,test_vectors,test_emails)
    
    #training error
    peggy2 = pegasos.pegasos()
    print "TRAINING ERROR:"
    miss = peggy2.pegasos_svm_test(weights,train_vectors, train_emails)
    
    
#---------END TESTING ON NEW DATA--------

#---------TRAINING ERROR-----------------
if "-trainerror" in sys.argv:
    vocabGenerator = vocabGen.vocabGen()
    vocab, emails = vocabGenerator.build_vocab('data/spam_train.txt', 4000)
    
    vecgen = vectorGen.vectorGen()
    vectors = vecgen.build_feature_arrays(vocab, emails)
    
    #training
    peggy = pegasos.pegasos()
    weights = peggy.pegasos_svm_train(vectors, emails, lambda_val, sv)
    
    #training error
    miss = peggy.pegasos_svm_test(weights,vectors,emails)
    print "   ->Training Error = " + str(float(miss) / float(len(vectors)))


else:
    print "Usage: python main.py algorithm [options]"
    print "Program Algorithms:"
    print "  -validate     : Run on training data. 4000 training, 1000 validation"
    print "  -test         : Run on test data. Use 5000 training set, 1000 test set"
    print "Program Options:"
    print "  -silent       : Only PEGASOS algorithm will output to terminal"
    print "  -l #          : Specify a lambda power. 2^#. Default is -5"
    print "  -sv           : Will find support vectors after training is complete"
    
    
    
    
    