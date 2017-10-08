import numpy as np
import regclass as rgr

E_Bayes=np.load('Error of Bayes.npy')
E_Robust=np.load('Error of Robust.npy')
E_Rgl=np.load('Error of Rgl.npy')
E_Lasso=np.load('Error of Lasso.npy')
E_LSE=np.load('Error of LSE.npy')

num_points = 44

rgr.Error_Plot(E_Bayes,'Error of Bayes',num_points)
rgr.Error_Plot(E_Robust,'Error of Robust',num_points)
rgr.Error_Plot(E_Rgl,'Error of Rgl',num_points)
rgr.Error_Plot(E_Lasso,'Error of Lasso',num_points)
rgr.Error_Plot(E_LSE,'Error of LSE',num_points)


