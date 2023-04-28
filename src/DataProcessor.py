# -*- coding: utf-8 -*-
import pandas as pd
from multiprocessing import Pool
import time
import numpy as np
from collections import defaultdict
import os

class DataProcessor:

    def __init__(self):
        path = os.getcwd()
        rates = '../include/ratings_small.csv'
        links = '../include/links_small.csv'
        moves = '../include/movies_metadata.csv'
        files = [moves,rates,links]

        #MULTI-THREADED (WORKS OUTSIDE OF SPYDER)
        with Pool(processes=3) as p:
            self.movies,self.rates,self.links = p.map(self.read_data,files)
            
        self.links.set_index("movieId",inplace=True)
        self.movies.set_index("id",inplace=True)
        self.uniq_movs = self.rates.movieId.unique()
        self.uniq_usrs = self.rates.userId.unique()
        self.newUserId = self.uniq_usrs.max()-1
        filtered_movies = []
        self.links['tmdbId']=self.links['tmdbId'].fillna(0)
        for movie in self.uniq_movs:
            if(self.links.at[movie,"tmdbId"]==0):
                filtered_movies.append(movie)
                print(movie)
        self.rates = self.rates[self.rates.movieId.isin(filtered_movies)==False]
        self.rates.drop('timestamp',axis=1, inplace=True)
        self.ratings = self.rates
        self.rates = self.rates.pivot(index='userId',columns='movieId',values='rating')
                       
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
    