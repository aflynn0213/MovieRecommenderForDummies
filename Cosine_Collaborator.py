# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 23:34:45 2023

@author: aflyn
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class Cosine_Collaborator:
    
    def __init__(self,df):
        self.df = df
        self.cos_mat = calc_similarity
        
    def calc_similarity(self):
        return pd.DataFrame(data=cosine_similarity(self.df),columns=self.df.index,index=self.df.index)
        
    def adjusted_cos(self):
        