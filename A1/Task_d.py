import regclass as rgr
import numpy as np
import scipy.io as scio
import random as rd
import matplotlib.pyplot as plt

data=scio.loadmat('poly_data.mat')
polyx=data['polyx']#1*100
polyy=data['polyy']#100*1
sampx=data['sampx']#1*50
sampy=data['sampy']#50*1
thtrue=data['thtrue']#6*1

origin_sampy=np.zeros((50,1))
for i in range(50):
    origin_sampy[i,0]=sampy[i,0]

New_sampy=rgr.Add_Outliers(sampx,sampy,5)
dim_Fea=5 #Feature dimension
variance=5 #variance of data
hyperparam={'rgl':1.8,'lasso':1,'Bayes':1}
Fea_Train=rgr.PolyFea(sampx,dim_Fea)
Fea_Test=rgr.PolyFea(polyx,dim_Fea)

Result=rgr.Task_b(Fea_Train,sampx,sampy,Fea_Test,polyx,polyy,variance,hyperparam,origin_sampy)
np.save('Task_d_Result',Result)


print Result
