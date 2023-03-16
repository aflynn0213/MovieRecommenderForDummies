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
            self.movies,self.rates,self.links = p.map(self.read_data,files)
            
        self.links.set_index("movieId",inplace=True)
        self.movies.set_index("id",inplace=True)
        self.uniq_movs = self.rates.movieId.unique()
        self.uniq_usrs = self.rates.userId.unique()
        
    def rand_movie_rater(self):
        movie_count = 1
        movie_dict = {}
        val_rates = [1,1.5,2,2.5,3,3.5,4,4.5,5,6]
        
        while(movie_count <= 20):
            print("\nMOVIE #"+str(movie_count))
            title = "INVALID"
            while (title=="INVALID"):
                rando = np.random.random_integers(0,len(self.uniq_movs)-1)
                rando = self.uniq_movs[rando]
                tmdb = self.moviedId_tmdbId_map(rando)
                title = self.fetch_title(tmdb)
            
            print(title)
            rating = float(input("RATING: \n6 for next"))
            if (rating not in val_rates):
                print("TRY AGAIN INVALID OPTION\n")
            elif(rating==6):
                print("OKAY DISPLAYING NEW MOVIE\n")
            else:
                movie_dict[title]=rating
                movie_count+=1
        #new_row = pd.DataFrame(movie_dict)
        
        
                
    def moviedId_tmdbId_map(self,mov_id):
        temp = self.links
        tmdb = str(int(temp.at[mov_id,"tmdbId"]))
        return tmdb
        
    def fetch_title(self,_id):
        temp = self.movies
        valid = _id in temp.index
        return (temp.at[_id,"original_title"] if valid else "INVALID")
    
    
    def read_data(self,x):
        if x == 'movies_metadata.csv':
            return (pd.read_csv(x,usecols=["id","original_language","original_title"],dtype={"id":"string","original_title":"string"}) [lambda m: m["original_language"]=="en"])
        else:
            return pd.read_csv(x)
