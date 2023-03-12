#!/usr/bin/python3
import pandas as pd
from multiprocessing import Pool
import numpy as np


def read_data(x):
    return pd.read_csv(x)

def gradient(A,U,M,stepsize,maxiter,tolerance=1e-02):
    
    eps = 2.2204e-14 #minimum step size for gradient descent

    loss = 10000
    stepsize = float(stepsize)
    err = np.zeros((A.shape[0],A.shape[1]))
    
    for i in range(maxiter):
        for j in range(len(A)):
            for k in range(len(A[0])):
                if A[j][k]!=0:
                    err[j][k]=A[j][k]-np.dot(U[j,:],M[:,k])
                    for f in range(10):    
                        U[j][f]=U[j][f]+stepsize*(2*err[j][k]*M[f][k]-.02*U[j][f])
                        M[f][k]=M[f][k]+stepsize*(2*err[j][k]*U[j][f]-.02*M[f][k])
        diff = 0
        for j in range(len(A)):
            for k in range(len(A[0])):
                if A[j][k]!=0:
                    diff += (A[j][k]-np.dot(U[j,:],M[:,k]))**2
                    for f in range(10):
                        diff+=(.01*(pow(U[j][f],2)+pow(M[f][k],2)))
        if (diff<0.01):
            break
        
        return U, M.T
    
if __name__ == '__main__':
    valid_val = True
    
    while(valid_val):
        print("WOULD THE USER LIKE THE FULL DATABASE OR THE LITE VERSION: ")
        opt = int(input("1 - FULL\n 2 - LITE (RECOMMENDED)\n"))
        if (opt != 1 and opt != 2):
            print("INVALID VALUE! TRY AGAIN\n")
            continue
        else:
            rates = 'ratings_small.csv' if opt==2 else 'ratings.csv'
            valid_val = False
    files = ['movies_metadata.csv','credits.csv',rates]
    
    #MULTI-THREADED (WORKS OUTSIDE OF SPYDER)
    with Pool(2) as p:
        movs,creds,rats = p.map(read_data,files)
    
    rats = pd.read_csv(rates)
 
    print("STEP 1 Just Read in CSVs")
    
    uniq_movs = rats.movieId.unique()
    uniq_usrs = rats.userId.unique()
    print("STEP 2 Filtered for unique movies and users in ratings csv ")

    
    rats.drop('timestamp',axis=1, inplace=True)
    print("STEP 3 Pivotting User Ratings csv to create SIM MATRIX\n")
    sim_mat = rats.pivot(index='userId',columns='movieId')
    sim_mat.fillna(0,inplace=True)
    
    print("STEP 4 Writing SIM Matrix to csv file")
    sim_mat.to_csv('sim_mat.csv',encoding='utf-8')
    #sim_mat.rename(columns = {sim_mat.columns : uniq_movs.sort()})
    
    #TODO: EMBEDDINGS????
    gd_mat = sim_mat.to_numpy(dtype=float)
    u_d = len(gd_mat)
    m_d= len(gd_mat[0])

    U = np.random.rand(u_d,10)
    M = np.random.rand(m_d,10)
    print("STEP 5 ABOUT TO RUN GRADIENT DESCENT ON USER AND MOVIE MATRICES")
    U,M = gradient(gd_mat, U, M.T, .0025, 5000)    
    
    print("STEP 6 DOTTING USER AND MOVIE MATRICES TO FORM RECOMMENDATION MATRIX")
    reco_mat = np.dot(U,M.T)
    
    #TURN INTO PANDAS DATAFRAME
    reco_mat = pd.DataFrame(reco_mat,index=sim_mat.index,columns=sim_mat.columns)
    print("STEP 7 WRITING RECOMMENDATION MATRIX TO CSV")
    reco_mat.to_csv('recommendation_matrix.csv',encoding='utf-8')
    
    
