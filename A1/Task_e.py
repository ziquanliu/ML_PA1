import regclass as rgr
import numpy as np
import scipy.io as scio
import random as rd
import matplotlib.pyplot as plt
import pickle

data=scio.loadmat('poly_data.mat')
polyx=data['polyx']#1*100
polyy=data['polyy']#100*1
sampx=data['sampx']#1*50
sampy=data['sampy']#50*1
thtrue=data['thtrue']#6*1


dim_Fea=10 #Feature dimension
variance=5 #variance of data
hyperparam={'rgl':1.8,'lasso':1,'Bayes':1}
Fea_Train=rgr.PolyFea(sampx,dim_Fea)
Fea_Test=rgr.PolyFea(polyx,dim_Fea)

Result=rgr.Task_b(Fea_Train,sampx,sampy,Fea_Test,polyx,polyy,variance,hyperparam,sampy)
f1=open('Task_e_Result.txt','wb')
pickle.dump(Result,f1)
f1.close()


print Result