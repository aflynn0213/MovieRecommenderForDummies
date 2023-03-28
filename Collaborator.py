# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 23:34:45 2023

@author: aflyn
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def calc_similarity(df,opt):
    norm = adjusted_cos(df,opt) 
    if opt == 1: 
        labels = df.index
    elif opt == 2: 
        labels = df.columns
        norm = norm.T
    temp = norm.fillna(0)
    ret1 = pd.DataFrame(data=cosine_similarity(temp),columns=labels,index=labels)
    ret2 = pd.DataFrame(data=norm,columns=df.columns,index=df.index)
    return ret2,ret1
def adjusted_cos(df,opt):
    if(opt==1):
        return df.sub(df.mean(axis=1,skipna=True),axis=0)
    elif(opt==2):
        return df.sub(df.mean(axis=0,skipna=True),axis=1)
    
def get_recommendations(A,cos,_id):
    not_seen = A.loc[_id,:]
    not_seen = np.where(not_seen>0,0,1)
    cosines = highest_cos(A,cos,_id)
    sim_users = A.loc[cosines.index,:]

    for index,row in sim_users.iterrows():
        sim_users.loc[index]=cosines.at[index]*sim_users.loc[index]
    #sim_users.replace(0, np.nan, inplace=True)
    movie_means = sim_users.mean()
    movie_means = movie_means.to_numpy()
    print(movie_means)
    user_mean = A.loc[_id].mean(skipna=True)
    A.loc[_id] = np.where(not_seen==1,movie_means,A.loc[_id])
    print(A.loc[_id])
    A.loc[_id]=A.loc[_id]+user_mean
    return A.loc[_id]

def highest_cos(A,cos,_id):
    df = cos.nlargest(26,_id)
    df = df.drop(df[_id].idxmax())
    df = df.loc[:,_id]
    return df

'''class Collaborator:
    
    def __init__(self,df,opt):
        self.df = df
        self.norm = self.adjusted_cos(opt)
        self.sim = self.calc_similarity(opt)
        
    def calc_similarity(self,opt):
        labels = self.df.index if opt == 1 else self.df.columns
        return pd.DataFrame(data=cosine_similarity(self.norm.T),columns=labels,index=labels)
        
    def adjusted_cos(self,opt):
        if(opt==1):
            return self.df.sub(self.df.mean(axis=1,skipna=True),axis=0)
        elif(opt==2):
            return self.df.sub(self.df.mean(axis=0,skipna=True),axis=1)
       
'''
   