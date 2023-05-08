from surprise import Reader, Dataset
#from surprise.model_selection import cross_validate
#from surprise.model_selection import train_test_split

from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise import SVD
from surprise.model_selection.search import GridSearchCV
import pandas as pd
from multiprocessing import Pool
from functools import partial
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import numpy as np

def run_benchmark(data):
    #algo = KNNWithZScore 
    algo = SVD
    print("RUNNING GRIDSEARCH")
    #params = {'k': [3,7],'sim_options': {'name': ['cosine', 'pearson_baseline'], 'user_based': [True], 'shrinkage':[25,50,100]},'verbose': [True]}
    params = { 'n_factors': [5, 10],'n_epochs': [10, 20], 'lr_all': [0.005, 0.07],'reg_all': [0.02, 0.05], 'verbose': [True]} 
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
    train_4 = pd.read_csv('benchmark_train_4.csv')
    train_5 = pd.read_csv('benchmark_train_5.csv')
    train_6 = pd.read_csv('benchmark_train_6.csv')
    
    #testing = pd.read_csv('benchmark_test.csv')
    data1 = Dataset.load_from_df(train,reader)
    data2 = Dataset.load_from_df(train_2,reader)
    data3 = Dataset.load_from_df(train_3, reader)
    data4 = Dataset.load_from_df(train_4, reader)
    data5 = Dataset.load_from_df(train_5, reader)
    data6 = Dataset.load_from_df(train_6, reader)
    
    data = [data1,data2,data3,data4,data5,data6]
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
    true_4 = defaultdict(list)
    
    estimated_1 = defaultdict(list)
    estimated_2_unrated = defaultdict(list)
    estimated_2_rated = defaultdict(list)
    estimated_3_unrated = defaultdict(list)
    estimated_3_rated = defaultdict(list)
    estimated_4_unrated = defaultdict(list)
    estimated_4_rated = defaultdict(list)
    estimated_5_unrated = defaultdict(list)
    estimated_6_unrated = defaultdict(list)
    
    #preds[0][1] should be empty
    #preds[x][0] is set of already rated
    #preds[x][1] is set of unrated movies

    #preds is a list of tuples, we only want the specified user and the estimated(predicted) rating (est)
   #############ALREADY RATED###############
    for uid,iid,true_r,est, _ in preds[0][0]:
        estimated_1[uid].append((iid,est))
        true_1[uid].append((iid,true_r))
    
    for uid, iid, true_r, est, _ in preds[1][0]:
        estimated_2_rated[uid].append((iid,est))
        true_2[uid].append((iid,true_r))
        
    for uid, iid, true_r, est, _ in preds[2][0]:
        estimated_3_rated[uid].append((iid,est))
        true_3[uid].append((iid,true_r))
    
    for uid, iid, true_r, est, _ in preds[3][0]:
        estimated_4_rated[uid].append((iid,est))
        true_4[uid].append((iid,true_r))        
    ###########ALREADY RATED################## 
    
    ##############UNRATED ESTIMATES#################
    for uid, iid, true_r, est, _ in preds[1][1]:
        estimated_2_unrated[iid].append(est)
    
    for uid, iid, true_r, est, _ in preds[2][1]:
        estimated_3_unrated[uid].append((iid,est))
    
    for uid, iid, true_r, est, _ in preds[3][1]:
        estimated_4_unrated[uid].append((iid,est))
        
    for uid, iid, true_r, est, _ in preds[4][1]:
        estimated_5_unrated[uid].append((iid,est))
        
    for uid, iid, true_r, est, _ in preds[5][1]:
        estimated_6_unrated[uid].append((iid,est))
    ############UNRATED ESTIMATES###################
    
    
    total = 0
    count = 0
    for mid,rates in estimated_2_unrated.items():
        print(mid)
        pred = estimated_2_unrated[mid][0]
        for i in true_1[15]:
            if mid == i[0]:
                actual = i[1]
                print("act")
                print(actual)
                break
        print("PRED")
        print(pred)
        total += (float(pred)-float(actual))**2 
        count += 1
    total = float(total)/count
    rmse2 = math.sqrt(total)
    print("ONE USER SPARSE VECTOR:")
    print(rmse2)
    
    total = 0
    count = 0
    for uid,_ in estimated_3_unrated.items():
        for est in estimated_3_unrated[uid]:
            for tv in true_1[uid]:
                if tv[0]==est[0]:
                    actual = tv[1]
                    break
            #print("ACT")
            #print(actual)
            pred = est[1]
            #print("PRED")
            #print(pred)
            total += (float(pred)-float(actual))**2 
            count += 1
    total = float(total)/count
    rmse3 = math.sqrt(total)
    print("THREE USERS SPARSE VECTORS:")
    print(rmse3) 
    
    for uid,_ in estimated_4_unrated.items():
        for est in estimated_4_unrated[uid]:
            for tv in true_1[uid]:
                if tv[0]==est[0]:
                    actual = tv[1]
                    break
            #print("ACT")
            #print(actual)
            pred = est[1]
            #print("PRED")
            #print(pred)
            total += (float(pred)-float(actual))**2 
            count += 1
    total = float(total)/count
    rmse4 = math.sqrt(total)
    print("SPARSE MATRIX:")
    print(rmse4) 
    
    for uid,_ in estimated_5_unrated.items():
        for est in estimated_5_unrated[uid]:
            for tv in true_1[uid]:
                if tv[0]==est[0]:
                    actual = tv[1]
                    break
            #print("ACT")
            #print(actual)
            pred = est[1]
            #print("PRED")
            #print(pred)
            total += (float(pred)-float(actual))**2 
            count += 1
    total = float(total)/count
    rmse5 = math.sqrt(total)
    print("SPARSER MATRIX:")
    print(rmse5) 
    
    for uid,_ in estimated_6_unrated.items():
        for est in estimated_6_unrated[uid]:
            for tv in true_1[uid]:
                if tv[0]==est[0]:
                    actual = tv[1]
                    break
            #print("ACT")
            #print(actual)
            pred = est[1]
            #print("PRED")
            #print(pred)
            total += (float(pred)-float(actual))**2 
            count += 1
    total = float(total)/count
    rmse6 = math.sqrt(total)
    print("SPARSEST MATRIX:")
    print(rmse6) 
    
    mat_size = 18*19
    x1 = len(estimated_2_unrated.items())
    x2 = 0
    for id_ in estimated_3_unrated.keys():
        x2 += len(estimated_3_unrated[id_])
    x3 = 0
    for id_ in estimated_4_unrated.keys():
        x3 += len(estimated_4_unrated[id_])
    x4 = 0
    for id_ in estimated_5_unrated.keys():
        x4 += len(estimated_5_unrated[id_])
    x5 = 0
    for id_ in estimated_6_unrated.keys():
        x5 += len(estimated_6_unrated[id_])
    
    x = np.divide([x1,x2,x3,x4,x5],mat_size)
    x = np.multiply(x,100)
    rmse = [rmse2,rmse3,rmse4,rmse5,rmse6]
    
    plt.plot(x,rmse)
    plt.xlabel('Sparsity')
    plt.ylabel('RMSE')
    
    
    