#!/usr/bin/python3
import pandas as pd
from multiprocessing import Pool
import numpy as np


#def read_data(x):
#    return pd.read_csv(x)

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
    #with Pool(2) as p:
     #   movs,creds,rats = p.map(read_data,files)
    
    rats = pd.read_csv(rates)
 
    print("STEP 1 Just Read in CSVs")
    
    uniq_movs = rats.movieId.unique()
    uniq_usrs = rats.userId.unique()
    print("STEP 2 Filtered for unique movies and users in ratings csv ")

    
    rats.drop('timestamp',axis=1, inplace=True)
    print("STEP 3 ABOUT TO SET VALUES IN SIM MATRIX\n")
    sim_mat = rats.pivot(index='userId',columns='movieId')
    sim_mat.to_csv('sim_mat.csv',encoding='utf-8')
    #sim_mat.rename(columns = {sim_mat.columns : uniq_movs.sort()})
    
    #TODO: EMBEDDINGS????
    
    
    