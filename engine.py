#!/usr/bin/python3
from DataProcessor import DataProcessor

from surprise import Reader, Dataset
#from surprise.model_selection import cross_validate
#from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise import SVD, SVDpp, BaselineOnly
from surprise.model_selection.search import GridSearchCV
import pandas as pd

class Engine:
    
    def __init__(self,opt):
        self.algorithm = opt
        print("STEP Reading in CSVs")
        self.dp = DataProcessor()
        print("STEP Pivotting User Ratings csv to create SIM MATRIX")
        self.reader = Reader(rating_scale=(1,5)) 
        self.data = Dataset.load_from_df(self.dp.ratings,self.reader)
        self.fullTrain = self.data.build_full_trainset()
        self.antiTest = self.fullTrain.build_anti_testset(fill=0)
        self.best_est = SVD() if opt==1 else KNNWithZScore()
        self.preds = self.data

    def run(self):
        if (self.algorithm == 1):
            self.run_mf()
        elif (self.algorithm == 2):
            self.run_kNN()
            
        return self.preds
        #self.common()
    
    def run_mf(self): 
        cv_df = []
        
        params = {'n_factors': [10,20,50],'lr_all':[0.0025,0.005],'reg_all': [0.02,0.01],'verbose':[True]}
        params_pp = {'n_factors': [10,20,50],'verbose':[True], 'cache_ratings':[True]}
         
        print("STEP Performing SVD GridSearchCV")
        algCV = GridSearchCV(SVD, param_grid=params,measures=["rmse","mae"],cv=4,refit=True,n_jobs=-1)   
        algCV.fit(self.data)
        cv_df.append(pd.DataFrame.from_dict(algCV.cv_results))
        print("RMSE scores for" + "SVD " + ": ")
        print(algCV.best_score["rmse"])
        print("With Parameters: ",algCV.best_params["rmse"])
        min_score = algCV.best_score["rmse"]
        self.best_est = algCV

        
        print("STEP Performing SVDpp GridSearchCV")
        algCV = GridSearchCV(SVDpp, param_grid=params_pp,measures=["rmse","mae"],cv=4,refit=True,n_jobs=-1)   
        algCV.fit(self.data)
        cv_df.append(pd.DataFrame.from_dict(algCV.cv_results))
        print("RMSE scores for" + "SVDpp " + ": ")
        print(algCV.best_score["rmse"])
        print("With Parameters: ",algCV.best_params["rmse"])
        tmp_score = algCV.best_score["rmse"]
        if (tmp_score<min_score):
            min_score = tmp_score
            self.best_est = algCV
        
        print("SVD GRIDSEARCH")
        cv_df[0].to_csv('svd_gridsearch.csv')
        
        print("SVDpp GRIDSEARCH")
        cv_df[1].to_csv('svdpp_gridsearch.csv')

        print("STEP Performing ALS and SGD Comparison")
        params = {'bsl_options':{'method': ['als','sgd']}}
        baselineCV = GridSearchCV(BaselineOnly, param_grid=params,measures=["rmse","mae"],cv=10,refit=True,n_jobs=-1)
        baselineCV.fit(self.data)
        cv_dfBase = pd.DataFrame.from_dict(baselineCV.cv_results)

        print("RMSE scores for" + "BaselineOnly" + ": ")
        print(baselineCV.best_score["rmse"])
        print("With Parameters: ",baselineCV.best_params["rmse"])
        cv_dfBase.to_csv('baseline_gridsearch.csv')
        
        baseline_score = baselineCV.best_score["rmse"]
        svd_score = min_score

        if (baseline_score <= svd_score):
            self.best_est = baselineCV

        print("BEST MATRIX FACTORIZATION ESTIMATOR IS: ")
        print(self.best_est.best_estimator["rmse"])
        print("WITH PARAMETERS: ")
        print(self.best_est.best_params["rmse"])
        
        self.best_est = self.best_est.best_estimator["rmse"]
        
        print("STEP Prediction unseen movies with best MF estimator")
        self.preds = self.best_est.fit(self.fullTrain).test(self.antiTest)        

    def run_kNN(self):
        params = {'k': [20, 40],'sim_options': {'name': ['pearson', 'cosine'],'min_support': [10,20],'user_based': [True]}}
        alg_objs = []
        
        for alg in [KNNWithZScore, KNNWithMeans, KNNBasic]:
            alg_objs.append(GridSearchCV(KNNWithZScore, param_grid=params,measures=["rmse","mae"],cv=4,refit=True,n_jobs=-1))
        
        print("STEP Performing Grid Search on KNNs")
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

        #tr, te = train_test_split(ratings, test_size=0.25)
        self.best_est = alg_objs[min_i].best_estimator["rmse"]
        print("BEST ESTIMATOR IS: ")
        print(self.best_est)
        print("WITH PARAMETERS OF: ")
        print(alg_objs[min_i].best_params["rmse"])
        self.best_est = KNNWithZScore(k=40,min_support=10,sim_options={"name":"pearson",})
        print("STEP Predicting unseen movies with best KNN estimator")
        self.preds = self.best_est.fit(self.fullTrain).test(self.antiTest)
        
    def run_new_user(self):
        self.data = Dataset.load_from_df(self.dp.ratings,self.reader)
        self.fullTrain = self.data.build_full_trainset()
        self.antiTest = self.fullTrain.build_anti_testset(fill=0)
        self.preds = self.best_est.fit(self.fullTrain).test(self.antiTest)
        

    # def common(self):
    #     user_opt = 0
    #     valid_opts = [1,2,3,4]

        
    
    #     while(user_opt != 3):
    #         print("\nWOULD YOU LIKE TO SEE A MOVIE RECOMMENDED FOR A CERTAIN USER OR YOURSELF?")
    #         user_opt = int(input("1)USER ID\n2)YOURSELF\n3)QUIT\n"))
    #         if (user_opt not in valid_opts):
    #             print("TRY AGAIN, VALID OPTIONS ARE 1-4\n")
    #             continue 
    #         elif (user_opt == 1):
    #             userID_input = int(input("USER ID: "))
    #             if (not self.fullTrain.knows_user(userID_input)):
    #                 print("INVALID USER ID, EXITING......")
    #             else:
    #                 self.dp.topTenPresentation(self.preds, userID_input)    
    #         elif (user_opt == 2):
    #             print("WE ARE GOING TO GENERATE A RANDOM LIST OF MOVIES\n")
    #             print("YOU HAVE THE OPTION OF RANKING THE MOVIE FROM 1-5 STARS\n")
    #             print("YOU CAN DO THIS INCREMENTS OF .5 STARS\n")
    #             print("IF YOU HAVEN'T SEE THE FILM THERE WILL BE A SKIP OPTION\n")
    #             print("PLEASE SKIP IF YOU HAVEN'T SEEN THE FILM!\n")
    #             print("PRESENTING FILMS NOW......")
    #             newRates,userId = self.dp.rand_movie_rater()
    #             self.dp.ratings = self.dp.ratings.append(newRates,ignore_index=True,sort=False)
    #             self.run_new_user()
    #             self.dp.topTenPresentation(self.preds,len(self.dp.ratings-1))     
    #         elif (user_opt == 3):
    #             print("THANK YOU FOR USING THE MOVIE RECOMMENDER\n")
    #             print("HOPE TO SEE YOU SOON!\n")
    #             return False

