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
    
    while(movie_count < 20):
        print("\nMOVIE #"+str(movie_count))
        title = "INVALID"
        while (title=="INVALID"):
            rando = np.random.random_integers(0,len(uniq_mov)-1)
            rando = uniq_mov[rando]
            tmdb = moviedId_tmdbId_map(links, rando)
            title = fetch_title(moves, tmdb)
        
        print(title)
        rating = float(input("RATING: \n6 for next"))
        if (rating not in val_rates):
            print("TRY AGAIN INVALID OPTION\n")
        elif(rating==6):
            print("OKAY DISPLAYING NEW MOVIE\n")
        else:
            movie_dict[title]=rating
            movie_count+=1
            
def moviedId_tmdbId_map(links,mov_id):
    temp = links.set_index("movieId")
    tmdb = str(int(temp.at[mov_id,"tmdbId"]))
    return tmdb
    
def fetch_title(movies,_id):
    temp = movies.set_index("id")
    valid = _id in temp.index
    return (temp.at[_id,"original_title"] if valid else "INVALID")