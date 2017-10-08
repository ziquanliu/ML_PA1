import regclass as rgr
import numpy as np
import scipy.io as scio
import pickle

data=scio.loadmat('poly_data.mat')
polyx=data['polyx']#1*100
polyy=data['polyy']#100*1
sampx=data['sampx']#1*50
sampy=data['sampy']#50*1
thtrue=data['thtrue']#6*1





#-------------------------------------------Training Part---------------------------------------------------------------
dim_Fea=5 #Feature dimension
variance=5 #variance of data
Fea_Trn=rgr.PolyFea(sampx,dim_Fea)


regress=rgr.Regressor(Fea_Trn,sampy,variance)

rgl_lambda=1.8 #hyperparameter of regulized LS
lasso_t=1    #hyperparameter of LASSO
Bay_alpha=1  #hyperparameter of Bayes

theta={'LSE':regress.LSE_Reg(),'Rgl':regress.Regul_Reg(rgl_lambda),'Las':regress.LASSO_Reg(lasso_t),
       'Rob':regress.Robust_Reg()}

Bayes_theta=regress.Bay_Reg(Bay_alpha)

#--------------------------------------------Testing Part---------------------------------------------------------------
Fea_Tst=rgr.PolyFea(polyx,dim_Fea)
Test_Pred={}
Train_Pred={}
for key in theta:
    Test_Pred[key] = rgr.nonBayes_Pred(theta[key],Fea_Tst)
    Train_Pred[key] = rgr.nonBayes_Pred(theta[key], Fea_Trn)


Bayes_Test_Pred = rgr.Bayes_Pred(Fea_Tst,Bayes_theta)
Bayes_Train_Pred = rgr.Bayes_Pred(Fea_Trn,Bayes_theta)


#---------------------------------------------Plot Part-----------------------------------------------------------------
#non Bayes
for key in theta:
    rgr.nonBayes_Plt(sampx ,sampy, Train_Pred[key],polyx,polyy,Test_Pred[key],key)


#Bayes
rgr.Bayes_Ply(sampx,sampy,Bayes_Train_Pred,polyx,polyy,Bayes_Test_Pred)

#----------------------------------------Error Analysis Part------------------------------------------------------------

Pred_Error=rgr.Pred_All_Err(polyy,Test_Pred,Bayes_Test_Pred)

f1=open('Task B Predict Error.txt','wb')
pickle.dump(Pred_Error,f1)
f1.close()





