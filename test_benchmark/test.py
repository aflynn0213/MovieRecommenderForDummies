from surprise import Reader, Dataset
#from surprise.model_selection import cross_validate
#from surprise.model_selection import train_test_split
#from surprise.prediction_algorithms.knns import KNNBasic
#from surprise.prediction_algorithms.knns import KNNWithMeans
#from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise import SVD#, SVDpp, BaselineOnly
from surprise.model_selection.search import GridSearchCV
import pandas as pd
from multiprocessing import Pool
from functools import partial
from collections import defaultdict
import math

def run_benchmark(data):
    algo = SVD
    print("RUNNING GRIDSEARCH")
    params = {'n_factors': [10,20,50],'lr_all':[0.0025,0.005],'reg_all': [0.02,0.01],'verbose':[True]}
    cv_obj = GridSearchCV(algo,params,measures=["mse","fcp"],cv=3,refit=True,n_jobs=-1,joblib_verbose=4)
    cv_obj.fit(data)
    best_est = cv_obj.best_estimator["mse"]
    train = data.build_full_trainset()
    test_set = train.build_testset()
    antitest_set = train.build_anti_testset()
    preds_rated = best_est.fit(train).test(test_set)
    preds_unrated = best_est.fit(train).test(antitest_set)
    return [preds_rated,preds_unrated]

if __name__ == '__main__':
    
    reader = Reader(rating_scale=(0.5,5)) 
    train = pd.read_csv('benchmark_train.csv')
    train_2 = pd.read_csv('benchmark_train_2.csv') 
    train_3 = pd.read_csv('benchmark_train_3.csv') 
    
    #testing = pd.read_csv('benchmark_test.csv')
    data1 = Dataset.load_from_df(train,reader)
    data2 = Dataset.load_from_df(train_2,reader)
    data3 = Dataset.load_from_df(train_3, reader)
    
    data = [data1,data2,data3]
    print("ABOUT TO POOL")
    pool = Pool()
    partial_cv = partial(run_benchmark)
    preds = pool.map(partial_cv,data)
    
    #End pool processes
    pool.close()
    pool.join()
    print("STEP finished multiprocressing")

    true_1 = defaultdict(list)
    true_2 = defaultdict(list)
    true_3 = defaultdict(list)
    
    estimated_1 = defaultdict(list)
    estimated_2_unrated = defaultdict(list)
    estimated_2_rated = defaultdict(list)
    estimated_3_unrated = defaultdict(list)
    estimated_3_rated = defaultdict(list)
    
    #preds[0][1] should be empty
    #preds[x][0] is set of already rated
    #preds[x][1] is set of unrated movies

    #preds is a list of tuples, we only want the specified user and the estimated(predicted) rating (est)
   #############ALREADY RATED###############
    for uid,iid,true_r,est, _ in preds[0][0]:
        if uid == 15:
            estimated_1[iid].append(est)
            true_1[iid].append(true_r)
    
    for uid, iid, true_r, est, _ in preds[1][0]:
        if uid == 15:
            estimated_2_rated[iid].append(est)
            true_2[iid].append(true_r)
    
        
    for uid, iid, true_r, est, _ in preds[2][0]:
        if uid == 15 or uid==73 or uid==452:
            estimated_3_rated[uid].append((iid,est))
            true_3[uid].append((iid,est))
    ###########ALREADY RATED################## 
    
    ##############UNRATED ESTIMATES#################
    for uid, iid, true_r, est, _ in preds[1][1]:
        if uid == 15:
            estimated_2_unrated[iid].append(est)
    
    for uid, iid, true_r, est, _ in preds[2][1]:
        if uid == 15 or uid==73 or uid==452:
            estimated_3_unrated[uid].append((iid,est))
            
    
    total = 0
    count = 0
    for mid,rates in estimated_2_unrated.items():
        pred = estimated_2_unrated[mid][0]
        actual = true_1[mid][0] 
        total += (float(pred)-float(actual))**2 
        count += 1
    total = float(total)/count
    rmse = math.sqrt(total)
    print("ONE USER SPARSE VECTOR:")
    print(rmse)
    
    total = 0
    count = 0
    for uid,_ in estimated_3_unrated.items():
        for it in estimated_3_unrated[uid]:
            print(it)
            print(it[0])
            for j in true_3[uid]:
                print(j)
                if j[0]==it[0]:
                    actual = j[1]
                    break
            print(actual)
            pred = it[1]
            total += (float(pred)-float(actual))**2 
            count += 1
    total = float(total)/count
    rmse = math.sqrt(total)
    print("THREE USERS SPARSE VECTORS:")
    print(rmse) 