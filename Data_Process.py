# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:31:18 2023

@author: aflyn
"""
import pandas as pd

class Data_Process:
    def __init__(self,option,files):
        
        #MULTI-THREADED (WORKS OUTSIDE OF SPYDER)
        with Pool(4) as p:
            self.movs,self.creds,self.rats,self.links = p.map(read_data,files)
        self.id_map = create_map()
        
    def create_map(self):
        temp_map = self.links
        temp_map.set_index('movieId',inplace=True)
        temp_map.drop('imbdId',axis=1,inplace=True)
        self.rats.drop('timestamp',axis=1, inplace=True)
        print("STEP 3 Pivotting User Ratings csv to create SIM MATRIX\n")
        
    def create_sim_mat(self):
        sim_mat = rats.pivot(index='userId',columns='movieId')
        sim_mat.fillna(0,inplace=True)
        
        
    def filter_foreign_movies(rats,links,movs):
    
        
    def read_data(x):
        if x == 'movies_metadata.csv':
            return (pd.read_csv(x,usecols=["id","original_language","original_title"],dtype={"id":"string","original_title":"string"}) [lambda m: m["original_language"]=="en"])
        else:
            return pd.read_csv(x)
    