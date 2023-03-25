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
    np.array(norm)
    return pd.DataFrame(data=cosine_similarity(norm),columns=labels,index=labels)
    
def adjusted_cos(df,opt):
    if(opt==1):
        return df.sub(df.mean(axis=1,skipna=True),axis=0)
    elif(opt==2):
        return df.sub(df.mean(axis=0,skipna=True),axis=1)
    
def compute_recommendations():
    x = np.where(x==0,calc_average()+user_avg,x)

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
   