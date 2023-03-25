# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 23:34:45 2023

@author: aflyn
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class Collaborator:
    
    def __init__(self,df):
        self.df = df
        self.sim_ = calc_similarity()
        
    def calc_similarity(self):
        return self.df.T.corr()
        
    #def adjusted_cos(self):
     #   mean = self.df.mean(axis=1)