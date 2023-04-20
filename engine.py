#!/usr/bin/python3
from DataProcessor import DataProcessor

from surprise import Reader, Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise import SVD, SVDpp, BaselineOnly
from surprise.model_selection.search import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Engine:
    
    def __init__(self,opt):
        self.algorithm = opt
        print("STEP Reading in CSVs")
        self.dp = DataProcessor()
        print("STEP Pivotting User Ratings csv to create SIM MATRIX")
        self.reader = reader = Reader(rating_scale=(1,5)) 
        self.data = Dataset.load_from_df(self.dp.ratings,self.reader)
        self.fullTrain = ratings.build_full_trainset()
        self.antiTest = self.fullTrain.build_anti_testset(fill=0)
        

    def run(self):
        if (self.algorithm == 1):
            self.run_mf()
        elif (self.algorithm == 2):
            self.run_kNN()
            
        self.common()
    
    def run_mf(self): 
        svds = []
        
        SVD_Alg = SVD(verbose=True)
        SVDpp_Alg = SVDpp(cache_ratings=True,verbose=True)
        params = {'n_factors': [10,20,50],'lr_all':[0.0025,0.005],'reg_all': [0.02,0.01]}
        
        min_score = 100000
        best_alg = SVD()
        print("STEP Performing SVD and SVDpp GridSearchCV")
        for alg in [SVD_Alg, SVDpp_Alg]:   
            algCV = GridSearchCV(alg, param_grid=params,measures=["rmse","mae"],cv=4,refit=True,n_jobs=-1)   
            algCV.fit(self.data)
            #cv_df = pd.DataFrame.from_dict(cv).mean(axis=0)
            #cv_df = cv_df.append(pd.Series([str(alg).split(" ")[0].split('.')[-1]], index=['Algorithm']))
            print("RMSE scores for" + str(alg) + ":")
            print(algCV.best_score["rmse"])
            print("With Parameters: ",algCV.best_params["rmse"])
            #print(cv_df)
            tmp_score = algCV.best_score["rmse"]
            if (tmp_score<min_score):
                min_score = tmp_score
                best_alg = algCV

        print("STEP Performing ALS and SGD Comparison")
        params = {'bsl_options':{'method': ['als','sgd']}}
        baselineCV = GridSearchCV(BaselineOnly, param_grid=params,measures=["rmse","mae"],cv=10,refit=True,n_jobs=-1)
        baselineCV.fit(self.data)
        
        print(baselineCV.best_score["rmse"])
        print(baselineCV.best_params["rmse"])
        
        baslin = baselineCV.best_score["rmse"]
        svd_alg = best_alg.best_score["rmse"]

        if (baslin < svd_alg):
            best_alg = baselineCV.best_estimator["rmse"]
        else:
            best_alg = best_alg.best_estimator["rmse"]

        predictions = best_alg.fit(self.fullTrain).test(self.antiTest)
        self.preds = predictions
        

    def run_kNN(self):
        params = {'k': [20, 40],'sim_options': {'name': ['pearson', 'cosine'],'min_support': [10,20],'user_based': [True]}}
        alg_objs = []
        
        for alg in [KNNWithZScore, KNNWithMeans, KNNBasic]:
            alg_objs.append(GridSearchCV(KNNWithZScore, param_grid=params,measures=["rmse","mae"],cv=4,refit=True,n_jobs=-1))
        
        alg_objs[0].fit(self.data)
        alg_objs[1].fit(self.data)
        alg_objs[2].fit(self.data)
        
        print(alg_objs[0].best_score["rmse"])
        print(alg_objs[1].best_score["rmse"])
        print(alg_objs[2].best_score["rmse"])

        min_score = 100000
        min_i = 0
        for i in range(len(alg_objs)):
            tmp_score = alg_objs[i].best_score["rmse"]
            if (tmp_score<min_score):
                min_score = tmp_score
                min_i = i

        algZ = alg_objs[0].best_estimator["rmse"]
        algM = alg_objs[1].best_estimator["rmse"]
        algB = alg_objs[2].best_estimator["rmse"]

        #tr, te = train_test_split(ratings, test_size=0.25)

        algZ.fit(self.fullTrain)
        algM.fit(self.fullTrain)
        algB.fit(self.fullTrain)

        predsB = algB.test(self.antiTest)
        predsM = algM.test(self.antiTest)
        predsZ = algZ.test(self.antiTest)

        predictions = alg_objs[min_i].best_estimator["rmse"].fit(self.fullTrain).test(self.antiTest)
        self.preds = predictions

    def common(self):
        user_opt = 0
        valid_opts = [1,2,3,4]

        while(user_opt != 3):
            print("\nWOULD YOU LIKE TO SEE A MOVIE RECOMMENDED FOR A CERTAIN USER OR YOURSELF?")
            user_opt = int(input("1)USER ID\n2)YOURSELF\n3)QUIT\n"))
            if (user_opt not in valid_opts):
                print("TRY AGAIN, VALID OPTIONS ARE 1-4\n")
                continue 
            elif (user_opt == 1):
                if self.algorithm == 1:
                    userID_input = int(input("USER ID: "))
                    if (self.fullTrain.knows_user(userID_input):
                        print("INVALID USER ID, EXITING......")
                    else:
                        self.dp.topTenPresentation(self.preds, userID_input)
                elif self.algorithm == 2:
                    userID_input = int(input("USER ID: "))
                    if (self.fullTrain.knows_user(userID_input):
                        print("INVALID USER ID, EXITING......")
                    else:
                        self.dp.topTenPresentation(self.preds, userID_input)    
             elif (user_opt == 2):
                print("WE ARE GOING TO GENERATE A RANDOM LIST OF MOVIES\n")
                print("YOU HAVE THE OPTION OF RANKING THE MOVIE FROM 1-5 STARS\n")
                print("YOU CAN DO THIS INCREMENTS OF .5 STARS\n")
                print("IF YOU HAVEN'T SEE THE FILM THERE WILL BE A SKIP OPTION\n")
                print("PLEASE SKIP IF YOU HAVEN'T SEEN THE FILM!\n")
                print("PRESENTING FILMS NOW......")
                newRates,userId = self.dp.rand_movie_rater()
                #self.dp.rates = self.dp.rates.append(newRates,ignore_index=True,sort=False)
                 
                if self.algorithm == 1:
                elif self.algorithm == 2:      
             elif (user_opt == 3):
                 print("THANK YOU FOR USING THE MOVIE RECOMMENDER\n")
                 print("HOPE TO SEE YOU SOON!\n")
                 return False

