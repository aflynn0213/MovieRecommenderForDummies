# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:59:43 2023

@author: aflyn
"""
import numpy as np
import pandas as pd

def rand_movie_gen(uniq_mov,moves,links):
    movie_count = 1
    movie_dict = {}
    val_rates = [1,1.5,2,2.5,3,3.5,4,4.5,5,6]
    
    links.set_index("movieId",inplace=True)
    moves.set_index("id",inplace=True)
    
    while(movie_count < 20):
        print("MOVIE #"+str(movie_count) +":\n")
        rando = np.random.random_integers(0,len(uniq_mov)-1)
        #print(rando)
        mov_id = uniq_mov[rando]
        print(mov_id)
        print("HERERERE")
        tmdb = int(links.at[mov_id,"tmdbId"])
        tmdb = str(tmdb)
        title = moves.at[tmdb,"original_title"]
        print(title)
        rating = float(input("RATING: \n6 for next"))
        if (rating not in val_rates):
            print("TRY AGAIN INVALID OPTION\n")
        elif(rating==6):
            print("OKAY DISPLAYING NEW MOVIE\n")
        else:
            movie_dict[title]=rating
            movie_count+=1
    