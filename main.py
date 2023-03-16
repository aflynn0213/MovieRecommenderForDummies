#!/usr/bin/python3
import pandas as pd
from multiprocessing import Pool
import numpy as np
<<<<<<< HEAD


=======
import time

#USER CREATED LIBRARIES
import DataProcessor as dp

def read_data(x):
    if x == 'movies_metadata.csv':
        return (pd.read_csv(x,usecols=["id","original_language","original_title"],dtype={"id":"string","original_title":"string"}) [lambda m: m["original_language"]=="en"])
    else:
        return pd.read_csv(x)

def filter_foreign_movies(rates,links,movs):
    
    
>>>>>>> 58ba12672020a63259ec9b37dcf2e4bf1cb30e2d
def gradient(A,U,M,stepsize,maxiter,tolerance=1e-02):
    
    eps = 2.2204e-14 #minimum step size for gradient descent

    loss = 10000
    stepsize = float(stepsize)
    err = np.zeros((A.shape[0],A.shape[1]))
<<<<<<< HEAD
    
    for i in range(maxiter):
        for j in range(len(A)):
            for k in range(len(A[0])):
                if A[j][k]!=0:
                    err[j][k]=A[j][k]-np.dot(U[j,:],M[:,k])
                    for f in range(10):    
                        U[j][f]=U[j][f]+stepsize*(2*err[j][k]*M[f][k]-.02*U[j][f])
                        M[f][k]=M[f][k]+stepsize*(2*err[j][k]*U[j][f]-.02*M[f][k])
=======
    us = len(A)
    ms = len(A[0])
    for i in range(maxiter):
        for j in range(us):
            for k in range(ms):
                if A[j][k]!=0:
                    err[j][k]=A[j][k]-np.dot(U[j,:],M[:,k])
                    for f in range(10):    
                        temp_U = U[j][f]+stepsize*(2*err[j][k]*M[f][k]-2/us*U[j][f]) 
                        temp_M = M[f][k]+stepsize*(2*err[j][k]*U[j][f]-2/ms*M[f][k]) 
                        U[j][f] = temp_U
                        M[f][k] = temp_M
>>>>>>> 58ba12672020a63259ec9b37dcf2e4bf1cb30e2d
        diff = 0
        for j in range(len(A)):
            for k in range(len(A[0])):
                if A[j][k]!=0:
<<<<<<< HEAD
                    diff += (A[j][k]-np.dot(U[j,:],M[:,k]))**2
                    for f in range(10):
                        diff+=(.01*(pow(U[j][f],2)+pow(M[f][k],2)))
=======
                    diff += 0.1*(A[j][k]-np.dot(U[j,:],M[:,k]))**2
                    for f in range(10):
                        diff+=(1/len(A)*(pow(U[j][f],2)+1/len(A[0])*pow(M[f][k],2)))
>>>>>>> 58ba12672020a63259ec9b37dcf2e4bf1cb30e2d
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
<<<<<<< HEAD
            links = 'links_small.csv' if opt==2 else 'links.csv'
            files = ['movies_metadata.csv','credits.csv',rates,links]
            dataProcessor = Data_Process(opt,files)
            valid_val = False
=======
            valid_val = False
    files = ['movies_metadata.csv','credits.csv',rates,'links_small.csv']
    
    #MULTI-THREADED (WORKS OUTSIDE OF SPYDER)
    with Pool(4) as p:
        movs,creds,rats,links = p.map(read_data,files)
    
    rats = pd.read_csv(rates)
 
    print("STEP 1 Just Read in CSVs")
>>>>>>> 58ba12672020a63259ec9b37dcf2e4bf1cb30e2d
    
    uniq_movs = rats.movieId.unique()
    #uniq_usrs = rats.userId.unique()
    #print("STEP 2 Filtered for unique movies and users in ratings csv ")

<<<<<<< HEAD

    #sim_mat.rename(columns = {sim_mat.columns : uniq_movs.sort()})
=======
    
    rats.drop('timestamp',axis=1, inplace=True)
    print("STEP 3 Pivotting User Ratings csv to create SIM MATRIX\n")
    sim_mat = rats.pivot(index='userId',columns='movieId',values='rating')
    sim_mat.fillna(0,inplace=True)
>>>>>>> 58ba12672020a63259ec9b37dcf2e4bf1cb30e2d
    
    #TODO: EMBEDDINGS????
    gd_mat = sim_mat.to_numpy(dtype=float)
    u_d = len(gd_mat)
    m_d= len(gd_mat[0])

    U = np.random.rand(u_d,10)
    M = np.random.rand(m_d,10)
    print("STEP 4 ABOUT TO RUN GRADIENT DESCENT ON USER AND MOVIE MATRICES")
<<<<<<< HEAD
    U,M = gradient(gd_mat, U, M.T, .0025, 5000)    
=======
    #U,M = gradient(gd_mat, U, M.T, .0025, 5000)    
>>>>>>> 58ba12672020a63259ec9b37dcf2e4bf1cb30e2d
    
    print("STEP 5 DOTTING USER AND MOVIE MATRICES TO FORM RECOMMENDATION MATRIX")
    reco_mat = np.dot(U,M.T)
    
    #TURN INTO PANDAS DATAFRAME
    reco_mat = pd.DataFrame(reco_mat,index=sim_mat.index,columns=sim_mat.columns)

    print("STEP 6 Writing to csv files")
<<<<<<< HEAD
    sim_mat.to_csv('sim_mat.csv',encoding='utf-8')
    reco_mat.to_csv('recommendation_matrix.csv',encoding ='utf-8')
    orig_mat = np.where(gd_mat != 0, 1, 0)
=======
    #sim_mat.to_csv('sim_mat.csv',encoding='utf-8')
    #reco_mat.to_csv('recommendation_matrix.csv',encoding ='utf-8')
    #orig_mat = np.where(gd_mat != 0, 1, 0)
>>>>>>> 58ba12672020a63259ec9b37dcf2e4bf1cb30e2d
    
    user_opt = 0
    valid_opts = [1,2,3]
    while(user_opt not in valid_opts):
        print("\nWOULD YOU LIKE TO SEE A MOVIE RECOMMENDED FOR A CERTAIN USER OR YOURSELF?")
        user_opt = int(input("1)USER ID\n2)YOURSELF\n3)QUIT\n"))
        if (user_opt not in valid_opts):
            print("TRY AGAIN, VALID OPTIONS ARE 1-3\n")
<<<<<<< HEAD
    if (user_opt == 1):
        print("PATH NOT PROGRAMMED YET, TBD\n")
    elif (user_opt == 2):
        from rand_movie_gen import rand_movie_gen
=======
    
    if (user_opt == 1):
        userID_input = int(input("USER ID: "))
        if (userID_input not in reco_mat.index):
            print("INVALID USER ID, EXITING......")
        else:    
            temp_row = reco_mat.loc[userID_input]
            top10 = temp_row.nlargest(10)
            print("RESULTS.......")
            count = 1
            for top in top10.index:
                title = dp.fetch_title(movs, dp.moviedId_tmdbId_map(links, top))
                time.sleep(2.5)
                print(str(count)+") "+title)
                count+=1
        
    elif (user_opt == 2):
>>>>>>> 58ba12672020a63259ec9b37dcf2e4bf1cb30e2d
        print("WE ARE GOING TO GENERATE A RANDOM LIST OF MOVIES\n")
        print("YOU HAVE THE OPTION OF RANKING THE MOVIE FROM 1-5 STARS\n")
        print("YOU CAN DO THIS INCREMENTS OF .5 STARS\n")
        print("IF YOU HAVEN'T SEE THE FILM THERE WILL BE A SKIP OPTION\n")
        print("PLEASE SKIP IF YOU HAVEN'T SEEN THE FILM!\n")
        print("PRESENTING FILMS NOW......")
<<<<<<< HEAD
        rand_movie_gen(uniq_movs,movs,links)
=======
        dp.rand_movie_gen(uniq_movs,movs,links)
        
>>>>>>> 58ba12672020a63259ec9b37dcf2e4bf1cb30e2d
    elif (user_opt == 3):
        print("THANK YOU FOR USING THE MOVIE RECOMMENDER\n")
        print("HOPE TO SEE YOU SOON!\n")

