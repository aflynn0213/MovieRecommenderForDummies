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
<<<<<<< HEAD
=======
from DataProcessor import DataProcessor
>>>>>>> jarrods_local

class Engine:
    
    def __init__(self,opt):
        self.algorithm = opt
        print("STEP Reading in CSVs")
        self.dp = DataProcessor()
        print("STEP Pivotting User Ratings csv to create SIM MATRIX")
        self.reco_mat = np.zeros(1)
        self.cos = np.zeros(1)
        
    def run(self):
        if (self.algorithm == 1):
            self.run_mf()
        elif (self.algorithm == 2):
            self.run_kNN()
            
        self.common()
    
    def run_mf(self): 
        A = self.dp.ratings
        reader = Reader(rating_scale=(1,5))      
        ratings = Dataset.load_from_df(A, reader)
        scores = []

        reader = Reader(rating_scale=(1,5))
        ratings = Dataset.load_from_df(A[['userId','movieId','rating']], reader) 

        for alg in [SVDpp(cache_ratings=True,init_mean=2.5,verbose=True)]:
            #params = {'n_epochs': [5,10],'lr_all':[0.001,0.005],'reg_all': [0.2,0.6]}
# =============================================================================
#             algCV = GridSearchCV(alg, param_grid=params,measures=["rmse","mae"],cv=4,refit=True,n_jobs=-1)
#             algCV.fit(ratings)
#             print(algCV.best_score["rmse"])
#             alg_best = algCV.best_estimator["rmse"]
#             train, test = train_test_split(ratings, test_size=0.25)
#             alg_best.fit(train)
# =============================================================================
            cv = cross_validate(alg, ratings, measures=['RMSE','MAE'], cv=4, verbose=True,n_jobs=-1)
            cv_df = pd.DataFrame.from_dict(cv).mean(axis=0)
            cv_df = cv_df.append(pd.Series([str(alg).split(" ")[0].split('.')[-1]], index=['Algorithm']))
            scores.append(cv_df)
            print("RMSE scores for" + str(alg) + ":")
            print(scores)
            #pred = alg_best.test(test)
            #print(pred)

        params = {'bsl_options':{'method': ['als','sgd']}}
        #baseline = BaselineOnly(params)
        baselineCV = GridSearchCV(BaselineOnly, param_grid=params,measures=["rmse","mae"],cv=10,refit=True,n_jobs=-1)

        
        baselineCV.fit(ratings)
        
        print(baselineCV.best_score["rmse"])
        
        algA = baselineCV.best_estimator["rmse"]
        
        train, test = train_test_split(ratings, test_size=0.25)
        
        algA.fit(train)
        
        predB= algA.test(test)
        #print(predB)

        #self.reco_mat = algA
        #print("RMSE for Baseline ALS")
        #accuracy.rmse(predict)
        

    def run_kNN(self):
        A = self.dp.ratings
        reader = Reader(rating_scale=(1,5))
        ratings = Dataset.load_from_df(A, reader=reader) 
        params = {'k': [20, 40],'sim_options': {'name': ['pearson', 'cosine'],'min_support': [10,20],'user_based': [False]}}
        
        knnZcv = GridSearchCV(KNNWithZScore, param_grid=params,measures=["rmse","mae"],cv=4,refit=True,n_jobs=-1)
        knnMcv = GridSearchCV(KNNWithMeans, param_grid=params,measures=["rmse","mae"],cv=4,refit=True,n_jobs=-1)
        knnBcv = GridSearchCV(KNNBasic, param_grid=params,measures=["rmse","mae"],cv=4,refit=True,n_jobs=-1)
        
        knnZcv.fit(ratings)
        knnMcv.fit(ratings)
        knnBcv.fit(ratings)
        
        print(knnZcv.best_score["rmse"])
        print(knnMcv.best_score["rmse"])
        print(knnBcv.best_score["rmse"])

        algZ = knnZcv.best_estimator["rmse"]
        algM = knnMcv.best_estimator["rmse"]
        algB = knnBcv.best_estimator["rmse"]
        
        tr = ratings.build_full_trainset()
        te = tr.build_anti_testset(fill=0)

        #tr, te = train_test_split(ratings, test_size=0.25)

        algZ.fit(tr)#fullTrain)
        algM.fit(tr)#fullTrain)
        algB.fit(tr)#fullTrain)

        predsB = algB.test(te)
        predsM = algM.test(te)
        predsZ = algZ.test(te)
        
        print(predsB)
        print(predsM)
        print(predsZ)

        self.reco_mat = algZ

    def common(self):
        user_opt = 0
        valid_opts = [1,2,3,4]
# =============================================================================
#         while(user_opt != 3):
#             print("\nWOULD YOU LIKE TO SEE A MOVIE RECOMMENDED FOR A CERTAIN USER OR YOURSELF?")
#             user_opt = int(input("1)USER ID\n2)YOURSELF\n3)QUIT\n"))
#             if (user_opt not in valid_opts):
#                 print("TRY AGAIN, VALID OPTIONS ARE 1-4\n")
#                 continue 
#             
#             elif (user_opt == 1):
#                 if self.algorithm == 1:
#                     userID_input = int(input("USER ID: "))
#                     #if (userID_input not in self.reco_mat.index):
#                     #    print("INVALID USER ID, EXITING......")
#                     #else:
#                         #self.dp.topTenPresentation(self.reco_mat, userID_input)
#                 elif self.algorithm == 2:
#                     #userID_input = int(input("USER ID: "))
#                     #if (userID_input not in self.reco_mat.index):
#                     #    print("INVALID USER ID, EXITING......")
#                     #else:
#                     #    self.reco_mat.loc[userID_input] = cb.get_recommendations(self.reco_mat,self.cos,userID_input)
#                     #    self.dp.topTenPresentation(self.reco_mat, userID_input)
#    
#             elif (user_opt == 2):
#                 print("WE ARE GOING TO GENERATE A RANDOM LIST OF MOVIES\n")
#                 print("YOU HAVE THE OPTION OF RANKING THE MOVIE FROM 1-5 STARS\n")
#                 print("YOU CAN DO THIS INCREMENTS OF .5 STARS\n")
#                 print("IF YOU HAVEN'T SEE THE FILM THERE WILL BE A SKIP OPTION\n")
#                 print("PLEASE SKIP IF YOU HAVEN'T SEEN THE FILM!\n")
#                 print("PRESENTING FILMS NOW......")
#                 newRates,userId = self.dp.rand_movie_rater()
#                 self.dp.rates = self.dp.rates.append(newRates,ignore_index=True,sort=False)
#                 
#                 if self.algorithm == 1:
#                     #A = self.dp.rates.fillna(0)
#                     #U,M,error = mf.gradient(A,features=25)
#                     #A = np.dot(U,M)
#                     #self.reco_mat = pd.DataFrame(A,index=self.dp.rates.index,columns=self.dp.rates.columns)
#                     #self.dp.topTenPresentation(self.reco_mat, userId)
#                     elif self.algorithm == 2:
#                     #self.reco_mat,self.cos = cb.calc_similarity(self.dp.rates,1)
#                     #self.reco_mat.loc[userId] = cb.get_recommendations(self.reco_mat,self.cos,userId)
#                     #self.dp.topTenPresentation(self.reco_mat, userId)
#                     
#             elif (user_opt == 3):
#                 print("THANK YOU FOR USING THE MOVIE RECOMMENDER\n")
#                 print("HOPE TO SEE YOU SOON!\n")
#                 return False
# 
# =============================================================================
