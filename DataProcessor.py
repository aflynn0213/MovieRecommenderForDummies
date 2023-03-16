# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:59:43 2023

@author: aflyn
"""
import numpy as np
import pandas as pd
from multiprocessing import Pool

class DataProcessor:

    def __init__(self,size):
        rates = 'ratings_small.csv' if size==2 else 'ratings.csv'
        links = 'links_small.csv' if size==2 else 'links.csv'
        files = ['movies_metadata.csv',rates,links]

        #MULTI-THREADED (WORKS OUTSIDE OF SPYDER)
        with Pool(3) as p:
            self.movies,self.rats,self.links = p.map(self.read_data,files)
            
        
    def rand_movie_gen(self,uniq_mov,moves,links):
        movie_count = 1
        movie_dict = {}
        val_rates = [1,1.5,2,2.5,3,3.5,4,4.5,5,6]
        
        while(movie_count < 20):
            print("\nMOVIE #"+str(movie_count))
            title = "INVALID"
            while (title=="INVALID"):
                rando = np.random.random_integers(0,len(uniq_mov)-1)
                rando = uniq_mov[rando]
                tmdb = moviedId_tmdbId_map(links, rando)
                title = fetch_title(moves, tmdb)
            
            print(title)
            rating = float(input("RATING: \n6 for next"))
            if (rating not in val_rates):
                print("TRY AGAIN INVALID OPTION\n")
            elif(rating==6):
                print("OKAY DISPLAYING NEW MOVIE\n")
            else:
                movie_dict[title]=rating
                movie_count+=1
                
    def moviedId_tmdbId_map(self,links,mov_id):
        temp = links.set_index("movieId")
        tmdb = str(int(temp.at[mov_id,"tmdbId"]))
        return tmdb
        
    def fetch_title(self,movies,_id):
        temp = movies.set_index("id")
        valid = _id in temp.index
        return (temp.at[_id,"original_title"] if valid else "INVALID")
    
    
    def read_data(self,x):
        if x == 'movies_metadata.csv':
            return (pd.read_csv(x,usecols=["id","original_language","original_title"],dtype={"id":"string","original_title":"string"}) [lambda m: m["original_language"]=="en"])
        else:
            return pd.read_csv(x)
