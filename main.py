#!/usr/bin/python3
import pandas as pd
import numpy as np
import time

#USER CREATED LIBRARIES
from DataProcessor import DataProcessor
#from MatrixFactorizer import MatrixFactorizer
import MatrixFactorizer as mf

def topTenPresentation(dp,recoMat,userId):
    temp_row = recoMat.loc[userId]
    top10 = temp_row.nlargest(10)
    print("RESULTS.......")
    count = 1
    for film in top10.index:
        title = dp.fetch_title(dp.moviedId_tmdbId_map(film))
        time.sleep(2.5)
        print(str(count)+") "+title)
        count+=1
  
if __name__ == '__main__':
    valid_val = True
    
    while(valid_val):
        print("WOULD THE USER LIKE THE FULL DATABASE OR THE LITE VERSION: ")
        opt = int(input("1 - FULL\n2 - LITE (RECOMMENDED)\n"))
        if (opt != 1 and opt != 2):
            print("INVALID VALUE! TRY AGAIN")
            continue
        else:
            dp = DataProcessor(opt)
            valid_val = False
 
    print("STEP 1 Just Read in CSVs")
    
    
    print("STEP 2 Pivotting User Ratings csv to create SIM MATRIX")
    sim_mat = dp.rates.pivot(index='userId',columns='movieId',values='rating')
    sim_mat.fillna(0,inplace=True)

    print("STEP 3 ABOUT TO RUN GRADIENT DESCENT ON USER AND MOVIE MATRICES")
    U,M = mf.gradient_handler(sim_mat)   
    
    print("STEP 4 DOTTING USER AND MOVIE MATRICES TO FORM RECOMMENDATION MATRIX")
    reco_mat = np.dot(U,M.T)

    print("STEP 5 Writing to csv files")
    #TURN INTO PANDAS DATAFRAME
    reco_mat = pd.DataFrame(reco_mat,index=sim_mat.index,columns=sim_mat.columns)
    #sim_mat.to_csv('sim_mat.csv',encoding='utf-8')
    #reco_mat.to_csv('recommendation_matrix.csv',encoding ='utf-8')
    #orig_mat = np.where(gd_mat != 0, 1, 0)
    
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
            topTenPresentation(dp, reco_mat, userID_input)
        
    elif (user_opt == 2):
        print("WE ARE GOING TO GENERATE A RANDOM LIST OF MOVIES\n")
        print("YOU HAVE THE OPTION OF RANKING THE MOVIE FROM 1-5 STARS\n")
        print("YOU CAN DO THIS INCREMENTS OF .5 STARS\n")
        print("IF YOU HAVEN'T SEE THE FILM THERE WILL BE A SKIP OPTION\n")
        print("PLEASE SKIP IF YOU HAVEN'T SEEN THE FILM!\n")
        print("PRESENTING FILMS NOW......")
        newRates,userId = dp.rand_movie_rater()
        sim_mat = sim_mat.append(newRates,ignore_index=True,sort=False)
        sim_mat.fillna(0,inplace=True)
        U,M = mf.gradient_handler(sim_mat) 
        reco_mat = np.dot(U,M.T)
        reco_mat = pd.DataFrame(reco_mat,index=sim_mat.index,columns=sim_mat.columns)
        topTenPresentation(dp, reco_mat, userId)
        
    elif (user_opt == 3):
        print("THANK YOU FOR USING THE MOVIE RECOMMENDER\n")
        print("HOPE TO SEE YOU SOON!\n")

