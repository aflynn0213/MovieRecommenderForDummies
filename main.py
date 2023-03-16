#!/usr/bin/python3
import pandas as pd
import numpy as np
import time

#USER CREATED LIBRARIES
from DataProcessor import DataProcessor
from MatrixFactorizer import MatrixFactorizer
    
if __name__ == '__main__':
    valid_val = True
    
    while(valid_val):
        print("WOULD THE USER LIKE THE FULL DATABASE OR THE LITE VERSION: ")
        opt = int(input("1 - FULL\n 2 - LITE (RECOMMENDED)\n"))
        if (opt != 1 and opt != 2):
            print("INVALID VALUE! TRY AGAIN\n")
            continue
        else:
            dp = DataProcessor(opt)
            valid_val = False
 
    print("STEP 1 Just Read in CSVs")
    uniq_movs = rats.movieId.unique()
    #uniq_usrs = rats.userId.unique()
    
    print("STEP 3 Pivotting User Ratings csv to create SIM MATRIX\n")
    rats.drop('timestamp',axis=1, inplace=True)
    sim_mat = rats.pivot(index='userId',columns='movieId',values='rating')
    sim_mat.fillna(0,inplace=True)

    
    print("STEP 4 ABOUT TO RUN GRADIENT DESCENT ON USER AND MOVIE MATRICES")
    #TURN SIM_MAT INTO NUMPY FOR MATRIX FACTORIZER OPERATIONS
    gd_mat = sim_mat.to_numpy(dtype=float)
    #INTIALIZING MATRIX FACTORIZER OBJECT
    mf = MatrixFactorizer()
    mf.gradient(gd_mat,.0025, 5000)    
    
    print("STEP 5 DOTTING USER AND MOVIE MATRICES TO FORM RECOMMENDATION MATRIX")
    reco_mat = np.dot(mf.U,mf.M.T)
    
    print("STEP 6 Writing to csv files")
    #TURN INTO PANDAS DATAFRAME
    reco_mat = pd.DataFrame(reco_mat,index=sim_mat.index,columns=sim_mat.columns)
    sim_mat.to_csv('sim_mat.csv',encoding='utf-8')
    reco_mat.to_csv('recommendation_matrix.csv',encoding ='utf-8')
    orig_mat = np.where(gd_mat != 0, 1, 0)
    
    user_opt = 0
    valid_opts = [1,2,3]
    while(user_opt not in valid_opts):
        print("\nWOULD YOU LIKE TO SEE A MOVIE RECOMMENDED FOR A CERTAIN USER OR YOURSELF?")
        user_opt = int(input("1)USER ID\n2)YOURSELF\n3)QUIT\n"))
        if (user_opt not in valid_opts):
            print("TRY AGAIN, VALID OPTIONS ARE 1-3\n")

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
        print("WE ARE GOING TO GENERATE A RANDOM LIST OF MOVIES\n")
        print("YOU HAVE THE OPTION OF RANKING THE MOVIE FROM 1-5 STARS\n")
        print("YOU CAN DO THIS INCREMENTS OF .5 STARS\n")
        print("IF YOU HAVEN'T SEE THE FILM THERE WILL BE A SKIP OPTION\n")
        print("PLEASE SKIP IF YOU HAVEN'T SEEN THE FILM!\n")
        print("PRESENTING FILMS NOW......")
        dp.rand_movie_gen(uniq_movs,movs,links)

    elif (user_opt == 3):
        print("THANK YOU FOR USING THE MOVIE RECOMMENDER\n")
        print("HOPE TO SEE YOU SOON!\n")

