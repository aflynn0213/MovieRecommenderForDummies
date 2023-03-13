# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:59:43 2023

@author: aflyn
"""
import numpy as np

def rand_movie_gen(A,moves,links):
    movie_count = 1
    movie_dict = {}
    while(movie_count < 20):
        print("MOVIE #"+str(movie_count) +":\n")
        rando = np.random.random_integers(1,len(A.columns))
        mov_id = A.columns[rando]
        tmbd = links.loc[links['movieId'] == mov_id, 'tmdbId']
        mov_tit = moves.loc[moves['id'] == tmbd, 'original_title']
    
    