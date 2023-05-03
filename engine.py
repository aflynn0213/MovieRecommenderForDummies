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
        self.reader = Reader(rating_scale=(0.5,5)) 
        #Full data set reads in our user movie ratings 
        self.data = Dataset.load_from_df(self.dp.ratings,self.reader)
        #Instance of trainset class from surprise
        self.fullTrain = self.data.build_full_trainset()
        #Test set is a list of ratings that can be used in test method later
        #returns only the ratings that have an actual rating
        self.test_set = self.fullTrain.build_testset()
        #returns all the ratings not in the trainset (all the ratings that are unknown)
        #used in predictions as this will predict all the unknown ratings and therefore
        #when we sort by highest predictions in this set, it will already be narrowed down
        #the movies unseen by a given user and will not require us to do further filtering to
        #only generate unseen movies i.e. it handles the filtering for us since 
        #this test set is only unobserved user-movie interactions to begin with
        self.antiTest = self.fullTrain.build_anti_testset(fill=0)
        #METRICS option to provide a table of simple statistics on a handful of algorithms
        if algorithm == "METRICS":
            self.performance_df = self.performance_metrics()
        #All options aside from metrics (a specific algorithm was chosen from index.html)
        else:
            self.best_est = self.hyperparameter_tuning()
            self.preds = self.best_est.fit(self.fullTrain).test(self.antiTest)
     
    #takes the given algorithm and runs gridsearch cv in order to find the best estimator 
    #and then returns this best estimator
    def hyperparameter_tuning(self):
        print("STEP Picking Algorithm: ")
        print(self.algorithm)
        if self.algorithm == "SVD":
            algo = SVD
            params = {'n_factors': [10,20,50],'lr_all':[0.0025,0.005],'reg_all': [0.02,0.01],'verbose':[True]}
        elif self.algorithm == "SVDpp":
            algo = SVDpp
            params = {'n_factors': [10,20,50], 'lr_all':[0.0025,0.005],'reg_all': [0.02,0.01],'verbose':[True], 'cache_ratings':[True]}
        elif self.algorithm == "ALS":
            algo = BaselineOnly
            params = {'bsl_options':{'method': ['als'],'reg_i':[10,15],'reg_u':[15,20],'n_epochs':[10,20]}}
        elif self.algorithm == "SGD":
            algo = BaselineOnly
            params = {'bsl_options':{'method': ['sgd'],'reg':[.02,.05],'learning_rate':[.005,.01,.02],'n_epochs':[15,20]}}
        elif self.algorithm == "KNNZ":
            algo = KNNWithZScore
            params = {'k': [20, 40], 'sim_options': {'name': ['pearson_baseline', 'cosine'],'shrinkage':[100,75,50],'min_support': [3],'user_based': [True]},'verbose':[True]}
        elif self.algorithm == "KNNM":
            algo = KNNWithMeans
            params = {'k': [20, 40], 'sim_options': {'name': ['pearson_baseline', 'cosine'],'shrinkage':[100,75,50],'min_support': [3],'user_based': [True]},'verbose':[True]}
        elif self.algorithm == "KNN":
            algo = KNNBasic
            params = {'k': [20, 40], 'sim_options': {'name': ['pearson_baseline', 'cosine'],'shrinkage':[100,75,50],'min_support': [3],'user_based': [True]},'verbose':[True]}    
        elif self.algorithm == "METRICS":
            self.performance_metrics()
        else:
            print("NOT A VALID ALGORITHM")
        
        cv_obj = GridSearchCV(algo,params,measures=["rmse","mae"],cv=3,refit=True,n_jobs=-1,joblib_verbose=4)
        cv_obj.fit(self.data)
        
        return cv_obj.best_estimator["rmse"]     
    
    #Used as the function in the multiprocessing pool when running METRICS
    #Performs gridsearch on large number of parameter combos and returns FCP, RMSE
    #and Best Parameters for each algorithm
    #Tests on whole dataset in order to best avoid overfitting
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
    
    #Creates parameter combos used in test_bestCV_alg and then 
    #returns dataframe with all the algorithms tested and their
    #statistics used to display in the metrics.html webpage
    def performance_metrics(self):
        print("STEP In performance metrics")
        u_mean = self.fullTrain.global_mean
        rates = [r[2] for r in self.fullTrain.all_ratings()]
        u_sd = statistics.stdev(rates)

        params = []
        #SVDpp, 
        algs = [SVD, BaselineOnly, BaselineOnly,KNNWithZScore] #, KNNWithMeans,KNNBasic]

        params1 = { 'n_factors': [20, 40],
                    'n_epochs': [20, 30], 
                    'lr_bu': [0.002, 0.005],
                    'lr_bi': [0.002, 0.005],
                    'lr_pu': [0.002, 0.005],
                    'lr_qi': [0.002, 0.005],
                    'reg_bu': [0.02, 0.01],
                    'reg_bi': [0.02, 0.01],
                    'reg_pu': [0.02, 0.01],
                    'reg_qi': [0.02, 0.01], 
                    'verbose': [True]}
        params.append(params1)

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
                     'sim_options': {'name': ['cosine', 'pearson_baseline'], 'user_based': [True], 'shrinkage':[25,50,100], 'min_support': [3]},
                    'verbose': [True]}
        params.append(paramsKnn)
        
        print("STEP About to start  multi-threaded gridsearch CV and test")
        #Creates tuples of (algorithm,parameter grid) combo since each
        #algorithm has one associated parameter grid with it
        #i.e. cv_args = [(SVD,params1),(BaselineOnly,params3),etc....
        cv_args = list(zip(algs,params))
        #instance of multiprocessing object
        pool = Pool()
        partial_cv = partial(self.test_bestCV_alg)
        #starmap---> [(1,2), (3, 4)] results in [partial(1,2), partial(3,4)]
        scores = pool.starmap(partial_cv,cv_args)
        
        #End pool processes
        pool.close()
        pool.join()
        print("STEP finished multiprocressing")

        #Rename since prior name was just BaselineOnly, BaselineOnly
        algs[1] = "ALS"
        algs[2] = "SGD"

        algs = [str(x) for x in algs]
        print("STEP Packing results from CV and testing into Dataframe")
        #scores from above is a list of list where each sublist was composed of
        #[FCP, RMSE, cv.best_params]
        df_data = { 'Algorithm': algs,
                    'FCP': [score[0] for score in scores],
                    'RMSE': [score[1] for score in scores],
                    'Best Parameter Set': [score[2] for score in scores]}
        
        #Create dataframe used for html presentation
        df = pd.DataFrame(df_data)
        df.set_index('Algorithm',inplace=True)

        return df
        
        
    def run_new_user(self,df):
        #concat at axis=0 adds new user vector as row in ratings 
        self.dp.ratings = pd.concat([self.dp.ratings,df],axis=0)
        #creates new dataset due to update to ratings with new user addition
        self.data = Dataset.load_from_df(self.dp.ratings,self.reader)
        #creates new trainset with new user entry
        self.fullTrain = self.data.build_full_trainset()
        #creates new antitestset to predict unseen movies
        self.antiTest = self.fullTrain.build_anti_testset(fill=0)
        #Uses best_estimator from EARLIER when the engine object was first initialized and 
        #tested against the entire data set in order to see how our previously cross valdiated
        #model generalized to new data.  It should be the best performing and most adaptive
        #to new data as this best estimator parameter combination achieved the highest score in 
        #the prior grid search cross validation step after the User chose an algorithm to initialize with
        #and because it was previously tested against the ENTIRE dataset and cross validated,
        #ideally it shouldn't be prone to overfitting that data and should generalize well.
        #Furthermore, we test it on the antitestset to limit our scope of predictions to only unseen
        #movies
        self.preds = self.best_est.fit(self.fullTrain).test(self.antiTest)
        