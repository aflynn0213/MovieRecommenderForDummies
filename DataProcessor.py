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
        
        #links is the mapping between the movie ids in our rating matrix
        #and the tmdbIds that appear in the movies_metadata csv file
        #and allow us to pull titles from to give user meaningful info
        self.links.set_index("movieId",inplace=True)
        #our movies dataframe only contains id (which is the tmdbId), title,
        #and original language. Setting ID to index
        self.movies.set_index("id",inplace=True)
        #Grab all unique movieIds from the ratings matrix 
        self.uniq_movs = self.rates.movieId.unique()
        #Find unique tmdbId corresponding to all movieIds that
        #actually have ratings and are useful for our data
        tmdb_ids = self.links.tmdbId.unique()
        #want the compliment (~) of NaN values since we need 
        #a valid tmdbId in order to fetch a title
        tmdb_ids = tmdb_ids[~np.isnan(tmdb_ids)]
        #want all tmdbIds as appear in movies metadata file 
        #in order to filter out movies whose tmdbIds don't show up
        #in this file
        metadata_ids = self.movies.index.unique()

        filtered_movies = []
        self.links['tmdbId']=self.links['tmdbId'].fillna(0)
        print("STEP Filtering out invalid tmdbIds")
        #filter out movies who don't have a tmdbId in the links dataframe
        for movie in self.uniq_movs:
            if(self.links.at[movie,"tmdbId"]==0):
                filtered_movies.append(movie)
                #DEBUG
                print(movie)
        #second filtering for movies whose tmdbIds don't appear in movies metadata        
        for tmdb in tmdb_ids:
            #tmdb_ids is a numpy array so we want to cut the .0 at the end by converting to int then string
            if str(int(tmdb)) not in metadata_ids:
                #the index of links is the movieId which are the ones in our user-movie rating matrix, so
                #for invalid tmdbIds we want to grab the correct "movieId" in order to filter out the 
                #movie from our ratings matrix
                Id = self.links.index[self.links.tmdbId == tmdb].tolist()
                #since list is only one element we can just grab first element
                Id = Id[0]
                filtered_movies.append(Id)
                #DEBUG
                print(Id)
                
        #Gets rid of rows where the movieId is in the filtered_movies list 
        #permanently removing movies that can't map to a title
        self.rates = self.rates[self.rates.movieId.isin(filtered_movies)==False]
        #Used in order to grab a random title in the main file to allow user to 
        #rate movies and establish a preference profile
        self.uniq_movs = self.rates.movieId.unique()
        #Timestamp is meaningless
        self.rates.drop('timestamp',axis=1, inplace=True)
        #ratings used as official user-movie rating matrix throughout
        self.ratings = self.rates
        #pivot and used to write to a csv for insights
        self.rates = self.rates.pivot(index='userId',columns='movieId',values='rating')
        self.rates.to_csv('filtered_rates.csv')
    
    #maps movieId from rating matrix to tmdbId used to find a title from movies metadata
    def moviedId_tmdbId_map(self,mov_id):
        tmdb = self.links.at[mov_id,"tmdbId"]
        return str(int(tmdb))
     
    #_id must be a valid tmdbId in order to fetch a title from movies metadata 
    def fetch_title(self,_id):
        valid = _id in self.movies.index
        return (self.movies.at[_id,"title"] if valid else "NOT IN DATABASE")
    
    #used to read in csvs and create pandas dataframes
    def read_data(self,x):
        #filters out everything aside from the mentioned columns
        if 'movies_metadata.csv' in x.lower():
            return (pd.read_csv(x,usecols=["id","original_language","title"],dtype={"id":"string","title":"string"})) #[lambda m: m["original_language"]=="en"])
        else:
            return pd.read_csv(x)
    