#!/usr/bin/env python3

import sys,math
import mnist_vectorGen,mnist_pegasos

vecGen = mnist_vectorGen.mnist_vectorGen()
data = vecGen.setup('data/mnist_train.txt')
#data will be returned as a list of tuples. tuple[0] is the class, tuple[1] is the feature vector
vecGen2 = mnist_vectorGen.mnist_vectorGen()
test_data = vecGen2.setup('data/mnist_test.txt')

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


lambda_val = math.pow(2,-3)
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

mult_weights = (weight0,weight1,weight2,weight3,weight4,weight5,weight6,weight7,weight8,weight9)

testPeggy = mnist_pegasos.pegasos()
miss = testPeggy.multiclass_pegasos_test(test_data, mult_weights, (0,1,2,3,4,5,6,7,8,9))