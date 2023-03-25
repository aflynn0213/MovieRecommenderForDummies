# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:16:26 2023

@author: aflyn
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#USER CREATED LIBRARIES
from DataProcessor import DataProcessor
import MatrixFactorizer as mf
import Collaborator as cb


class Engine:
    
    def __init__(self,opt):
        self.algorithm = opt
        print("STEP 1 Reading in CSVs")
        self.dp = self.get_size()
        print("STEP 2 Pivotting User Ratings csv to create SIM MATRIX")
        self.reco_mat = np.zeros(1)
        
    def get_size(self):
        while(1):
            print("WOULD THE USER LIKE THE FULL DATABASE OR THE LITE VERSION: ")
            size = int(input("1 - FULL\n2 - LITE (RECOMMENDED)\n"))
            if (size != 1 and size != 2):
                print("INVALID VALUE! TRY AGAIN")
                continue
            else:
                dp = DataProcessor(size)
                return dp
        
    def run(self):
        if (self.algorithm == 1):
            self.reco_mat = self.run_gd()
        elif (self.algorithm == 2):
            self.reco_mat = self.run_cosine()
        
        self.reco_mat = pd.DataFrame(self.reco_mat,index=self.dp.rates.index,columns=self.dp.rates.columns)
        self.common()
                    
    def run_cosine(self):
        opt = int(input("1)USER-USER \n2)ITEM-ITEM\n"))
        opt = opt if opt == 1 or opt ==2 else 1 
        mat = cb.calc_similarity(self.dp.rates,opt)
        mat.to_csv('CollaborativeFiltering.csv',encoding='utf-8')
        return mat
    
    def run_gd(self):
        print("STEP ABOUT TO RUN GRADIENT DESCENT ON USER AND MOVIE MATRICES")
        A = self.dp.rates
        U,M,err = mf.gradient_handler(A,20)
        
        print("STEP DOTTING USER AND MOVIE MATRICES TO FORM RECOMMENDATION MATRIX")
        reco_mat = np.dot(U,M)

        print("STEP Writing to csv files")
        #TURN INTO PANDAS DATAFRAME
        #self.dp.rates.to_csv('original_matrix.csv',encoding='utf-8')
        #self.reco_mat.to_csv('recommendation_matrix.csv',encoding ='utf-8')
        #orig_mat = np.where(gd_mat != 0, 1, 0)
        return reco_mat

    def common(self):
        user_opt = 0
        valid_opts = [1,2,3,4]
        while(user_opt != 3):
            print("\nWOULD YOU LIKE TO SEE A MOVIE RECOMMENDED FOR A CERTAIN USER OR YOURSELF?")
            user_opt = int(input("1)USER ID\n2)YOURSELF\n3)QUIT\n4)METRICS\n"))
            if (user_opt not in valid_opts):
                print("TRY AGAIN, VALID OPTIONS ARE 1-4\n")
                continue 
            
            elif (user_opt == 1):
                userID_input = int(input("USER ID: "))
                if (userID_input not in self.reco_mat.index):
                    print("INVALID USER ID, EXITING......")
                else:
                    self.dp.topTenPresentation(self.reco_mat, userID_input)
            
            elif (user_opt == 2):
                print("WE ARE GOING TO GENERATE A RANDOM LIST OF MOVIES\n")
                print("YOU HAVE THE OPTION OF RANKING THE MOVIE FROM 1-5 STARS\n")
                print("YOU CAN DO THIS INCREMENTS OF .5 STARS\n")
                print("IF YOU HAVEN'T SEE THE FILM THERE WILL BE A SKIP OPTION\n")
                print("PLEASE SKIP IF YOU HAVEN'T SEEN THE FILM!\n")
                print("PRESENTING FILMS NOW......")
                newRates,userId = self.dp.rand_movie_rater()
                self.dp.rates = self.dp.rates.append(newRates,ignore_index=True,sort=False)
                
                if self.algorithm == 1:
                    A = self.dp.rates.fillna(0)
                    U,M,error = mf.gradient_handler(A,25)
                    self.reco_mat = np.dot(U,M)
                    self.reco_mat = pd.DataFrame(self.reco_mat,index=self.dp.rates.index,columns=self.dp.rates.columns)
                    self.dp.topTenPresentation(self.reco_mat, userId)
                elif self.algorithm == 2:
                    print("TODO")
                    
            elif (user_opt == 3):
                print("THANK YOU FOR USING THE MOVIE RECOMMENDER\n")
                print("HOPE TO SEE YOU SOON!\n")
            
            elif (user_opt==4):
                feat_arr = [2,5,10,20,30,40,50]
                feat_dict = dict()
                for i in feat_arr:
                    print("Running Gradient Descent with " + str(i) + " features")
                    A = self.dp.rates
                    U,M,err = mf.gradient_handler(A, i)
                    feat_dict[i] = err
                ky = min(feat_dict)
                min_ky = min(feat_dict, key = lambda k:feat_dict[k])
                fig = plt.figure(figsize = (10, 5))
                plt.plot(feat_dict.keys(),feat_dict.values())
                plt.xlabel("# Features")
                plt.ylabel("Error")
                plt.title("Plot of Error vs Features")
                plt.show()
                
                print("Number of Features w/ Minimum Error: ", min_ky)
                print("Error by # of features", feat_dict)
