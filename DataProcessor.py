# -*- coding: utf-8 -*-
import pandas as pd
from multiprocessing import Pool
import time
import numpy as np
from collections import defaultdict
import os

class DataProcessor:

    def __init__(self):
        cwd=os.getcwd()
        rates = cwd+'\\include\\ratings_small.csv'
        links = cwd+'\\include\\links_small.csv'
        moves = cwd+'\\include\\movies_metadata.csv'
        files = [moves,rates,links]

        #MULTI-THREADED (WORKS OUTSIDE OF SPYDER)
        with Pool(processes=3) as p:
            self.movies,self.rates,self.links = p.map(self.read_data,files)
        
        self.links.set_index("movieId",inplace=True)
        self.movies.set_index("id",inplace=True)
        self.uniq_movs = self.rates.movieId.unique()
        tmdb_ids = self.links.tmdbId.unique()
        tmdb_ids = tmdb_ids[~np.isnan(tmdb_ids)]
        metadata_ids = self.movies.index.unique()
        filtered_movies = []
        self.links['tmdbId']=self.links['tmdbId'].fillna(0)
        
        for movie in self.uniq_movs:
            if(self.links.at[movie,"tmdbId"]==0):
                filtered_movies.append(movie)
                print(movie)
                
        for tmdb in tmdb_ids:
            if str(int(tmdb)) not in metadata_ids:
                Id = self.links.index[self.links.tmdbId == tmdb].tolist()
                Id = Id[0]
                filtered_movies.append(Id)
                print(Id)
                
            
        self.rates = self.rates[self.rates.movieId.isin(filtered_movies)==False]
        self.uniq_movs = self.rates.movieId.unique()
        self.rates.drop('timestamp',axis=1, inplace=True)
        self.ratings = self.rates
        self.rates = self.rates.pivot(index='userId',columns='movieId',values='rating')
        self.rates.to_csv('filtered_rates.csv')
               
    def moviedId_tmdbId_map(self,mov_id):
        temp = self.links
        tmdb = temp.at[mov_id,"tmdbId"]
        return str(int(tmdb))
        
    def fetch_title(self,_id):
        valid = _id in self.movies.index
        return (self.movies.at[_id,"title"] if valid else "NOT IN DATABASE")
     
    def read_data(self,x):
        if 'movies_metadata.csv' in x.lower():
            return (pd.read_csv(x,usecols=["id","original_language","title"],dtype={"id":"string","title":"string"})) #[lambda m: m["original_language"]=="en"])
        else:
            return pd.read_csv(x)
    