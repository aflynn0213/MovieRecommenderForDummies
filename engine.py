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
import statistics 
from surprise import accuracy
from multiprocessing import Pool
from functools import partial

class Engine:
    
    def __init__(self,algorithm):
        self.algorithm = algorithm
        print("STEP Reading in Data from CSVs")
        self.dp = DataProcessor()
        print("STEP Creating Training, Test, and AntiTest sets")
        self.reader = Reader(rating_scale=(1,5)) 
        self.data = Dataset.load_from_df(self.dp.ratings,self.reader)
        self.fullTrain = self.data.build_full_trainset()
        self.test_set = self.fullTrain.build_testset()
        self.antiTest = self.fullTrain.build_anti_testset(fill=0)
        if algorithm == "METRICS":
            self.performance_df = self.performance_metrics()
        else:
            self.best_est = self.hyperparameter_tuning()
            self.preds = self.best_est.fit(self.fullTrain).test(self.antiTest)
        
    def hyperparameter_tuning(self):
        print("STEP Picking Algorithm: ")
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
    
    def test_bestCV_alg(self,algo,params):
        gs_cv = GridSearchCV(algo,params,measures=['rmse'],cv=3,n_jobs=-1,joblib_verbose=2)
        gs_cv.fit(self.data)

        b_params = gs_cv.best_params["rmse"] 
        print(gs_cv.best_score["rmse"])
        print(b_params)
        best_est = gs_cv.best_estimator['rmse'].fit(self.fullTrain)
        preds = best_est.test(self.test_set)

        fcp_score = accuracy.fcp(preds,verbose=True)
        rmse_score = accuracy.rmse(preds,verbose=True)

        return [fcp_score,rmse_score,b_params]
        
    def performance_metrics(self):
        print("STEP In performance metrics")
        u_mean = self.fullTrain.global_mean
        rates = [r[2] for r in self.fullTrain.all_ratings()]
        u_sd = statistics.stdev(rates)

        params = []
        algs = [SVD, SVDpp, BaselineOnly, BaselineOnly,KNNWithZScore, KNNWithMeans,KNNBasic]

        params1 = { 'n_factors': [20, 40],
                    'n_epochs': [20, 30], 
                    'lr_all': [0.005, 0.07],
                    'reg_all': [0.02, 0.05], 
                    'verbose': [True]}
        params.append(params1)


        params2 = { 'n_factors': [20, 40],
                    'n_epochs': [20, 30], 
                    'lr_all': [0.005, 0.07],
                    'reg_all': [0.02, 0.05],
                    'verbose': [True],
                    'cache_ratings': [True]}
        params.append(params2)

        params3 = {'bsl_options': { 'method': ['als'],
                                    'n_epochs': [5, 10, 15],
                                    'reg_u': [10, 15, 20],
                                    'reg_i': [5, 10, 15]},
                                    'verbose': [True]}
        params.append(params3)

        params4 = { 'bsl_options': 
                    {   'method': ['sgd'],
                        'learning_rate': [.001, .025, .005],
                        'reg': [.02, .05, .1],
                        'n_epochs': [10, 20, 30]},
                        'verbose': [True]}
        params.append(params4)

        paramsKnn = {'k': [20, 40, 50],
                     'sim_options': [{'name': 'cosine', 'user_based': True, 'min_support': 3},
                                     {'name': 'pearson_baseline', 'user_based': True, 'min_support': 3, 'shrinkage': 25},
                                     {'name': 'pearson_baseline', 'user_based': True, 'min_support': 3, 'shrinkage': 50}],
                    'verbose': [True]}
        params.append(paramsKnn)
        params.append(paramsKnn)
        params.append(paramsKnn)
        
        print("STEP About to start  multi-threaded gridsearch CV and test")
        cv_args = list(zip(algs,params))
        pool = Pool()
        partial_cv = partial(self.test_bestCV_alg)
        scores = pool.starmap(partial_cv,cv_args)

        pool.close()
        pool.join()


        algs[2] = "ALS"
        algs[3] = "SGD"

        algs = [str(x) for x in algs]
        print("STEP Packing results from CV and testing into Dataframe")
        df_data = { 'Algorithm': algs,
                    'FCP': [score[0] for score in scores],
                    'RMSE': [score[1] for score in scores],
                    'Best Parameter Set': [score[2] for score in scores]}

        df = pd.DataFrame(df_data)
        df.set_index('Algorithm',inplace=True)

        return df
        
        
    def run_new_user(self,df):
        self.dp.ratings = pd.concat([self.dp.ratings,df],axis=0)
        self.data = Dataset.load_from_df(self.dp.ratings,self.reader)
        self.fullTrain = self.data.build_full_trainset()
        self.antiTest = self.fullTrain.build_anti_testset(fill=0)
        self.preds = self.best_est.fit(self.fullTrain).test(self.antiTest)
        