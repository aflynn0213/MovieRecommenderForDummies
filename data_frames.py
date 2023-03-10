#!/usr/bin/python3
import pandas as pd
from multiprocessing import Pool
import numpy as np


def read_data(x):
    return pd.read_csv(x)

if __name__ == '__main__':
    files = ['movies_metadata.csv','credits.csv','ratings.csv']
    #multithreaded processes
    #proc1 = multiprocessing.Process(target=read_movies) 
    #proc2 = multiprocessing.Process(target=read_credits)
    #proc3 = multiprocessing.Process(target=read_ratings)
    
    #start processes
    #proc1.start()
    #proc2.start()
    #proc3.start()
    
    #proc1.join()
    #proc2.join()
    #proc3.join()
    
    with Pool(5) as p:
        movs,creds,rats = p.map(read_data,files)
    
    print("Just Read in CSVs")
    
    uniq_movs = rats.movieId.unique()
    uniq_usrs = rats.userId.unique()
    print("Filtered for unique movies and users in ratings csv ")
    
    sim_mat = pd.DataFrame(np.zeros((len(uniq_usrs),len(uniq_movs)),np.int8),index=uniq_usrs,columns=uniq_movs)
    print(sim_mat.shape, "THIS IS THE EMPTY SIMILARITY MATRIX TO BE OPTIMIZED")
    
    #TODO: EMBEDDINGS????
    
    
    
    