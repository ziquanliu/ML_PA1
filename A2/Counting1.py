import regclass_A2 as rgr
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import pickle

data=scio.loadmat('count_data.mat')
testx=data['testx']
testy=data['testy']
trainx=data['trainx']
trainy=data['trainy']
ym=data['ym']
print 'number of test', testy.shape

#Assume the variance to be 1
var=1
hyperparam={'rgl':1.8,'lasso':1,'Bayes':1}


Pred_E=rgr.Counting(trainx,trainy,testx,testy,var,hyperparam)

f1=open('Predict Error.txt','wb')
pickle.dump(Pred_E,f1)
f1.close()


