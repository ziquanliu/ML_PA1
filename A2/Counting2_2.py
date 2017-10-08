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

var=1
hyperparam={'rgl':1.8,'lasso':1,'Bayes':1}
#try different features

def Counting_Feature_Cross(x):
    tempx=x.copy()
    num=x.shape[1]
    Feature=np.zeros((45,num))
    for i in range(num):
        ind=0
        for j in range(9):
            for k in range(j,9):
                Feature[ind,i]=tempx[j,i]*tempx[k,i]
                ind+=1
    return Feature

def Adding_Fea(x,n):
    tempx=x.copy()
    num=x.shape[1]
    Feature=np.zeros((9,num))
    for i in range(num):
        ind=0
        for j in range(9):
            Feature[ind,i]=tempx[j,i]*tempx[0,i]
            ind+=1
    return Feature


tempx=trainx.copy()
squ_train=tempx**2
temp_test=testx.copy()
squ_test=temp_test**2
cub_train=tempx**3
cub_test=temp_test**3
Fea_Train=np.row_stack((tempx,squ_train,cub_train))
Fea_Test=np.row_stack((temp_test,squ_test,cub_test))
Pred_E = rgr.Counting(Fea_Train, trainy, Fea_Test, testy, var, hyperparam)
f1=open('Counting2_2/Cube Feature Predict Error.txt','wb')
pickle.dump(Pred_E,f1)
f1.close()

#for i in range(9):
#    Added_F_Train=Adding_Fea(trainx,i)
#    Added_F_Test=Adding_Fea(testx,i)
#    Fea_Train=np.row_stack((trainx,Added_F_Train))
#    Fea_Test=np.row_stack((testx,Added_F_Test))
#    Abs_Pred_E = rgr.Counting(Fea_Train, trainy, Fea_Test, testy, var, hyperparam)['AbsErr']
#    f1=open('Add_Feature/Added Feature'+str(i)+'Predict Error.txt','wb')
#    pickle.dump(Abs_Pred_E,f1)
#    f1.close()






#f1=open('Feature Cross Predict Error.txt','wb')
#pickle.dump(Pred_E,f1)
#f1.close()






