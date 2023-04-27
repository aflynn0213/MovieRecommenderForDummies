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
from surprise.accuracy
import pandas as pd

class Engine:
    
    def __init__(self,algorithm):
        self.algorithm = algorithm
        print("STEP Reading in CSVs")
        self.dp = DataProcessor()
        print("STEP Pivotting User Ratings csv to create SIM MATRIX")
        self.reader = Reader(rating_scale=(1,5)) 
        self.data = Dataset.load_from_df(self.dp.ratings,self.reader)
        self.fullTrain = self.data.build_full_trainset()
        self.antiTest = self.fullTrain.build_anti_testset(fill=0)
        if algorithm == "METRICS":
            self.performance_metrics()
        else:
            self.best_est = self.hyperparameter_tuning()
            self.preds = self.best_est.fit(self.fullTrain).test(self.antiTest)
        
    def hyperparameter_tuning(self):
        print(self.algorithm)
        if self.algorithm == "SVD":
            algo = SVD
            params = {'n_factors': [10,20,50],'lr_all':[0.0025,0.005],'reg_all': [0.02,0.01],'verbose':[True]}
        elif self.algorithm == "SVDpp":
            algo = SVDpp
            params = {'n_factors': [10,20,50],'verbose':[True], 'cache_ratings':[True]}
        elif self.algorithm == "ALS":
            algo = BaselineOnly
            params = {'bsl_options':{'method': ['als'],'reg_i':[10,15],'reg_u':[15,20],'n_epochs':[10,20]}}
        elif self.algorithm == "SGD":
            algo = BaselineOnly
            params = {'bsl_options':{'method': ['sgd'],'reg':[.02,.05],'learning_rate':[.005,.01,.02],'n_epochs':[15,20]}}
        elif self.algorithm == "KNNZ":
            algo = KNNWithZScore
            params = {'sim_options': {'name': ['pearson', 'cosine'],'shrinkage':[100,75,50],'min_support': [7],'user_based': [True]},'verbose':[True]}
        elif self.algorithm == "KNNM":
            algo = KNNWithMeans
            params = {'sim_options': {'name': ['pearson', 'cosine'],'shrinkage':[100,75,50],'min_support': [7],'user_based': [True]},'verbose':[True]}
        elif self.algorithm == "KNN":
            algo = KNNBasic
            params = {'sim_options': {'name': ['pearson', 'cosine'],'shrinkage':[100,75,50],'min_support': [7],'user_based': [True]},'verbose':[True]}    
        elif self.algorithm == "METRICS":
            self.performance_metrics()
        else:
            print("NOT A VALID ALGORITHM")
        
        cv_obj = GridSearchCV(algo,params,measures=["rmse","mae"],cv=3,refit=True,n_jobs=-1,joblib_verbose=4)
        cv_obj.fit(self.data)
        
        return cv_obj.best_estimator["rmse"]      

        
    def performance_metrics(self):
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
        
        
        
    def run_new_user(self,df):
        self.dp.ratings = pd.concat([self.dp.ratings,df],axis=0)
        self.data = Dataset.load_from_df(self.dp.ratings,self.reader)
        self.fullTrain = self.data.build_full_trainset()
        self.antiTest = self.fullTrain.build_anti_testset(fill=0)
        self.preds = self.best_est.fit(self.fullTrain).test(self.antiTest)
        