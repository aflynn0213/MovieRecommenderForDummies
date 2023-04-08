# -*- coding: utf-8 -*-

import pandas as pd
from multiprocessing import Pool
import time
import numpy as np

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
        self.newUserId = self.uniq_usrs.max()-1
        self.rates.drop('timestamp',axis=1, inplace=True)
        self.rates = self.rates.pivot(index='userId',columns='movieId',values='rating')
        
        
    def rand_movie_rater(self):
        movie_count = 1
        movie_dict = {}
        val_rates = ['1','1.5','2','2.5','3','3.5','4','4.5','5','6']
        total_movies = 1
        
        while(movie_count<=20 and total_movies<=60):
            print("\nMOVIE #"+str(movie_count))
            title = "NOT IN DATABASE"
            while (title=="NOT IN DATABASE"):
                rando = np.random.random_integers(0,len(self.uniq_movs)-1)
                rando = self.uniq_movs[rando]
                tmdb = self.moviedId_tmdbId_map(rando)
                title = self.fetch_title(tmdb)
            
            print(title)
            rating = input("RATING: \n6 for next\n")
            if (rating not in val_rates):
                print("TRY AGAIN INVALID OPTION\n")
            elif(rating=='6'):
                print("OKAY DISPLAYING NEW MOVIE\n")
            else:
                movie_dict[rando]=[float(rating)]
                movie_count+=1
            total_movies += 1
            
        self.newUserId+=1
        new_row = pd.DataFrame(movie_dict,index=[self.newUserId])
        return new_row, self.newUserId
                
    def moviedId_tmdbId_map(self,mov_id):
        temp = self.links
        tmdb = temp.at[mov_id,"tmdbId"]
        return str(int(tmdb))
        
    def fetch_title(self,_id):
        valid = _id in self.movies.index
        return (self.movies.at[_id,"title"] if valid else "NOT IN DATABASE")
    
    
    def read_data(self,x):
        if x == 'movies_metadata.csv':
            return (pd.read_csv(x,usecols=["id","original_language","title"],dtype={"id":"string","title":"string"})) #[lambda m: m["original_language"]=="en"])
        else:
            return pd.read_csv(x)
        
    def topTenPresentation(self,recoMat,userId):
        temp_row = recoMat.loc[userId]
        top10 = temp_row.nlargest(10)
        print("RESULTS.......")
        count = 1
        for film in top10.index:
            title = self.fetch_title(self.moviedId_tmdbId_map(film))
            time.sleep(2.5)
            print(str(count)+") "+title)
            count+=1
