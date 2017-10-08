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

#print trainx.shape
var=1
hyperparam={'rgl':1.8,'lasso':1,'Bayes':1}
#try different feauture
#F1:x1,x2...x9,x1**2,x2**2...x9**2
tempx=trainx.copy()
squ_x=tempx**2
New_trainx=np.row_stack((trainx.copy(),squ_x))
temp_test=testx.copy()
squ_test=temp_test**2
New_testx=np.row_stack((testx.copy(),squ_test))
Pred_E=rgr.Counting(New_trainx,trainy,New_testx,testy,var,hyperparam)

print Pred_E

f1=open('Counting2_1/Predict Error.txt','wb')
pickle.dump(Pred_E,f1)
f1.close()