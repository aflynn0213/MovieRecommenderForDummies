# -*- coding: utf-8 -*-
import numpy as np
from surprise import Reader, Dataset, SVD, SVDpp, BaselineOnly
from surprise.model_selection import cross_validate
import pandas as pd

# =============================================================================
def gradient(A,features,stepsize=.005,maxiter=1500):
    results = []
    reader = Reader(rating_scale=(1,5))
    ratings_data = Dataset.load_from_df(A[['userId','movieId','rating']], reader) 
    for alg in [SVD(), SVDpp(), BaselineOnly()]:  
        cv_results = cross_validate(alg, ratings_data, measures=['RMSE','MAE'], cv=3, verbose=True)
        results_df = pd.DataFrame.from_dict(cv_results).mean(axis=0)
        results_df = results_df.append(pd.Series([str(alg).split('')[0].split('.')[-1]], index=['Algorithm']))
        results.append(results_df)
    print(results)
    
#     A = A.to_numpy(dtype=float)
#     users = len(A)
#     movies = len(A[0])
#     U = np.random.rand(users,features)
#     M = np.random.rand(features,movies)
#     for i in range(maxiter):
#         print(i)
#         for j in range(users):
#             for k in range(movies):
#                 if A[j][k]!=0:
#                     err=A[j][k]-np.dot(U[j,:],M[:,k])
#                     for f in range(features):
#                         temp_U = U[j][f]+stepsize*(2*err*M[f][k]-2/float(users)*U[j][f])
#                         temp_M = M[f][k]+stepsize*(2*err*U[j][f]-2/float(movies)*M[f][k])
#                         U[j][f] = temp_U
#                         M[f][k] = temp_M
#         diff = 0
#         for j in range(users):
#             for k in range(movies):
#                 if A[j][k]!=0:
#                     diff += 1/float(features)*(A[j][k]-np.dot(U[j,:],M[:,k]))**2
#                     for f in range(features):
#                         diff+=(1/users)*(pow(U[j][f],2))+(1/movies)*(pow(M[f][k],2))
#         if (diff<0.01):
#             break
# 
#     return U,M, diff
# =============================================================================


# =============================================================================
#     A=A.to_numpy(dtype=float)
#     users = len(A)
#     movies = len(A[0])
#     U = np.random.rand(users,features)
#     M = np.random.rand(features,movies)
#     
#     for i in range(maxiter):
#         err = np.where(A!=0,np.subtract(A,np.dot(U,M)),0)
#         count = np.sum(np.where(A!=0,1,0))
#         regU=(1/float(users))*np.sum(np.sum(U**2,axis=1))
#         regM=(1/float(movies))*np.sum(np.sum(M**2,axis=0))
#         loss = np.sum(err**2)/float(count)+regU+regM
#         gradU = -2*np.dot(err,M.T)+np.divide(U,float(users))
#         gradM = -2*np.dot(U.T,err)+np.divide(M,float(movies))
#         U = U + stepsize*gradU
#         M = M + stepsize*gradM
#     
#         if loss < 1:
#             print("FOUND LOCAL MIN")
#             break
#     
#     return U,M,loss
# =============================================================================


'''class MatrixFactorizer:

    def __init__(self,A):
        self.A = A
        u_d = len(A)
        m_d = len(A[0])
        self.U = np.random.rand(u_d,10)
        self.M = np.random.rand(m_d,10)

    def gradient(self,stepsize,maxiter,feats=10):

        eps = 2.2204e-14 #minimum step size for gradient descent

        loss = 10000
        stepsize = float(stepsize)

        A = self.A
        U = self.U
        M = self.M.T

        err = np.zeros((A.shape[0],A.shape[1]))
        us = len(A)
        ms = len(A[0])
        crdnl = 1/float(feats)

        for i in range(maxiter):
            for j in range(us):
                for k in range(ms):
                    if A[j][k]!=0:
                        err[j][k]=A[j][k]-np.dot(U[j,:],M[:,k])
                        for f in range(feats):
                            temp_U = U[j][f]+stepsize*(2*err[j][k]*M[f][k]-.02*U[j][f])
                            temp_M = M[f][k]+stepsize*(2*err[j][k]*U[j][f]-.02*M[f][k])
                            U[j][f] = temp_U
                            M[f][k] = temp_M

            diff = 0
            for j in range(len(A)):
                for k in range(len(A[0])):
                    if A[j][k]!=0:
                        diff += crdnl*(A[j][k]-np.dot(U[j,:],M[:,k]))**2
                        for f in range(feats):
                            diff+=.01*(pow(U[j][f],2)+pow(M[f][k],2))

            if (diff<0.01):
                break

        self.A = A
        self.U = U
        self.M = M.T
'''