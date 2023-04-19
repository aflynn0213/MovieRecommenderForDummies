#USER CREATED LIBRARIES
from DataProcessor import DataProcessor
import MatrixFactorizer as mf
import Collaborator as cb
from surprise import Reader, Dataset, SVD, SVDpp, BaselineOnly
from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from recommenders.utils.timer import Timer
#from recommenders.models.surprise.surprise_utils import predict
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise.model_selection.search import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from surprise.model_selection import train_test_split

class Engine:
    
    def __init__(self,opt):
        self.algorithm = opt
        print("STEP Reading in CSVs")
        self.dp = self.get_size()
        print("STEP Pivotting User Ratings csv to create SIM MATRIX")
        self.reco_mat = np.zeros(1)
        self.cos = np.zeros(1)
        
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
            self.run_gd()
        elif (self.algorithm == 2):
            self.run_cosine()
            
        self.common()
    
    def run_gd(self):
        print("STEP ABOUT TO RUN SVD ON USER AND MOVIE MATRICES")
        A = self.dp.ratings
        A= A.fillna(0)
        results = []
        reader = Reader(rating_scale=(1,5))
        ratings_data = Dataset.load_from_df(A[['userId','movieId','rating']], reader) 
        for alg in [SVD(), SVDpp(), BaselineOnly()]:  
            cv_results = cross_validate(alg, ratings_data, measures=['RMSE','MAE'], cv=3, verbose=True)
            results_df = pd.DataFrame.from_dict(cv_results).mean(axis=0)
            results_df = results_df.append(pd.Series([str(alg).split('')[0].split('.')[-1]], index=['Algorithm']))
            results.append(results_df)
        print(results)
        scores = []
        reader = Reader(rating_scale=(1,5))
        ratings = Dataset.load_from_df(A[['userId','movieId','rating']], reader) 
        cv_scores = cross_validate(SVD(), ratings, measures=['RMSE','MAE'], cv=3, verbose=True)
        scores_df = pd.DataFrame.from_dict(cv_scores).mean(axis=0)
        scores_df = scores_df.append(pd.Series([str(SVD()).split(" ")[0].split('.')[-1]], index=['Algorithm']))
        scores.append(scores_df)
        print("Error scores for SVD")
        print(scores)
        
        #Train and Predict with SVD using ALS
        #options = {}
        print('Train & Predict with SVD using Alternating Least Squares')
        tr, te = train_test_split(ratings, test_size=0.25)
        svd = SVD(random_state=0, n_factors=250, n_epochs=30, verbose=True)
        #with Timer() as train_time:
        svd.fit(tr)
        #print("Took {} seconds for training.".format(train_time.interval))
        #cross_validate(svd, ratings, measures=['RMSE'], cv=3, verbose=False)
       
        pred = svd.fit(tr).test(te)
        #predictions = predict(svd, te, usercol='userID', itemcol='itemID')
        #predictions.head()
        print("Accuracy Score using RMSE")
        print(accuracy_score.rmse(pred))
        #U,M,err = mf.gradient(A,features=20)
        
        
        #print("STEP DOTTING USER AND MOVIE MATRICES TO FORM RECOMMENDATION MATRIX")
        #self.reco_mat = np.dot(U,M)
        #self.reco_mat = pd.DataFrame(self.reco_mat,index=self.dp.rates.index,columns=self.dp.rates.columns)
        
        #print("STEP Writing to csv files")
        #TURN INTO PANDAS DATAFRAME
        #self.dp.rates.to_csv('original_matrix.csv',encoding='utf-8')
        #self.reco_mat.to_csv('recommendation_matrix.csv',encoding ='utf-8')
        #orig_mat = np.where(gd_mat != 0, 1, 0)
 
    
    def run_cosine(self):
        A = self.dp.ratings
        reader = Reader(rating_scale=(1,5))
        ratings = Dataset.load_from_df(A, reader=reader) 
        params = {'k': [10, 20, 40],'sim_options': {'name': ['pearson', 'cosine'],'min_support': [5, 10,20],'user_based': [True,False]}}
        tr, te = train_test_split(ratings, test_size=0.25)
        knnZcv = GridSearchCV(KNNWithZScore, param_grid=params,measures=["rmse","mae"],cv=None,refit=True,return_train_measures=True,n_jobs=-1)
        knnZcv.fit(ratings)
        print(knnZcv.best_params["rmse"],knnZcv.best_score["rmse"],knnZcv.cv_results)
       
        preds = knnZcv.test(te)
        self.reco_mat,self.cos = cb.calc_similarity(self.dp.rates,1)
        #self.cos.to_csv('CollaborativeFiltering.csv',encoding='utf-8')

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
                if self.algorithm == 1:
                    userID_input = int(input("USER ID: "))
                    if (userID_input not in self.reco_mat.index):
                        print("INVALID USER ID, EXITING......")
                    else:
                        
                        self.dp.topTenPresentation(self.reco_mat, userID_input)
                elif self.algorithm == 2:
                    userID_input = int(input("USER ID: "))
                    if (userID_input not in self.reco_mat.index):
                        print("INVALID USER ID, EXITING......")
                    else:
                        self.reco_mat.loc[userID_input] = cb.get_recommendations(self.reco_mat,self.cos,userID_input)
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
                    U,M,error = mf.gradient(A,features=25)
                    A = np.dot(U,M)
                    self.reco_mat = pd.DataFrame(A,index=self.dp.rates.index,columns=self.dp.rates.columns)
                    self.dp.topTenPresentation(self.reco_mat, userId)
                elif self.algorithm == 2:
                    self.reco_mat,self.cos = cb.calc_similarity(self.dp.rates,1)
                    self.reco_mat.loc[userId] = cb.get_recommendations(self.reco_mat,self.cos,userId)
                    self.dp.topTenPresentation(self.reco_mat, userId)
                    
            elif (user_opt == 3):
                print("THANK YOU FOR USING THE MOVIE RECOMMENDER\n")
                print("HOPE TO SEE YOU SOON!\n")
                return False
            
            elif (user_opt==4):
                
                
                feat_arr = [2,5,10,20,30,40,50]
                feat_dict = dict()
                for i in feat_arr:
                    print("Running Gradient Descent with " + str(i) + " features")
                    A = self.dp.rates
                    U,M,err = mf.gradient(A, features=i)
                    feat_dict[i] = err
                ky = min(feat_dict)
                min_ky = min(feat_dict, key = lambda k:feat_dict[k])
                plt.plot(feat_dict.keys(),feat_dict.values())
                plt.xlabel("# Features")
                plt.ylabel("Error")
                plt.title("Plot of Error vs Features")
                plt.show()
                
                print("Number of Features w/ Minimum Error: ", min_ky)
                print("Error by # of features", feat_dict)
