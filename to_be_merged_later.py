from surprise import Reader, Dataset
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise import SVD, SVDpp, BaselineOnly
from surprise.model_selection.search import GridSearchCV
from surprise import accuracy

from multiprocessing import Pool
from functools import partial

def test_bestCV_alg(algo,params):
    gs_cv = GridSearchCV(algo,params,measures=['rmse'],cv=3,n_jobs=-1,joblib_verbose=2)
    gs_cv.fit(self.data)

    b_params = gs.best_params["rmse"] 
    print(gs.best_score["rmse"])
    print(b_params)
    best_est = gs_cv.best_estimator['rmse'].fit(self.fullTrain)
    preds = best_est.test(self.test_set)

    fcp_score = accuracy.fcp(preds,verbose=True)
    rmse_score = accuracy.rmse(preds,verbose=True)

    return [fcp_score,rmse_score,b_params]

self.dp = DataProcessor()
self.reader = Reader(rating_scale=(1,5)) 
self.data = Dataset.load_from_df(self.dp.ratings,self.reader)
self.fullTrain = self.data.build_full_trainset()
self.test_set = self.fullTrain.build_testset()
self.antiTest = self.fullTrain.build_anti_testset(fill=0)

u_mean = self.fullTrain.global_mean
u_sd = self.fullTrain.global_std_dev

fcp_scores = []
rmse_scores = []
best_params = []
params = []
algs = [SVD,SVDpp,BaselineOnly,BaselineOnly,KNNWithZScore, KNNWithMeans,KNNBasic]

params1 = { 'n_factors': [20, 40],
            'n_epochs': [20, 30], 
            'lr_all': [0.005, 0.07],
            'reg_all': [0.02, 0.05], 
            'init_mean': [u_mean],
            'init_std_dev': [u_sd],
            'verbose': [True]}
params.append(params1)


params2 = { 'n_factors': [20, 40],
            'n_epochs': [20, 30], 
            'lr_all': [0.005, 0.07],
            'reg_all': [0.02, 0.05], 
            'init_mean': [u_mean],
            'init_std_dev': [u_sd],
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
                'verbose': [False]}
params.append(params4)

paramsKnn = { 'k': [20, 40, 50],
            'sim_options': [{'name': 'cosine', 'user_based': True, 'min_support': 3},
                            {'name': 'cosine', 'user_based': False, 'min_support': 3},
                            {'name': 'pearson_baseline', 'user_based': True, 'min_support': 3, 'shrinkage': 25},
                            {'name': 'pearson_baseline', 'user_based': False, 'min_support': 3, 'shrinkage': 25},
                            {'name': 'pearson_baseline', 'user_based': True, 'min_support': 3, 'shrinkage': 50},
                            {'name': 'pearson_baseline', 'user_based': False, 'min_support': 3, 'shrinkage': 50}],
            'verbose': [True]}
params.append(paramsKnn)
params.append(paramsKnn)
params.append(paramsKnn)

cv_args = list(zip(algs,params))
pool = Pool()
partial_cv = partial(test_bestCV_alg)
scores = pool.starmap(partial_cv,cv_args)

pool.close()
pool.join()


algs[2] = ALS
algs[3] = SGD

algs = [str(x) for x in algs]

df_data = { 'Algorithm': algs,
            'FCP': [score[0] for score in scores],
            'RMSE': [score[1] for score in scores],
            'Best Parameter Set': [score[2] for score in scores]}

df = pd.DataFrame(df_data)
df.set_index('Algorithm',inplace=True)
