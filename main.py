from flask import Flask, render_template, request, redirect, url_for
from collections import defaultdict
from engine import Engine
from DataProcessor import DataProcessor
import random

import pandas as pd

main = Flask(__name__)

movie_list = []
engines = []

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        algorithm = request.form['algorithm']
        valid_algs = ["SVD","SVDpp","ALS","SGD","KNNZ","KNNM","KNN"]
        if algorithm in valid_algs:
            return redirect(url_for('select_user', algorithm=algorithm))
        elif algorithm == "METRICS":
            return redirect(url_for('metrics'))
    return render_template('index.html')

@main.route('/select-user/<algorithm>', methods=['GET', 'POST'])
def select_user(algorithm):
    if request.method == 'POST':
        user_id = request.form['user_id']
        if 'rate_movie' in request.form:
            eng = Engine(algorithm)
            engines.append(eng)
            return redirect(url_for('rate_movie', algorithm=algorithm))
        return redirect(url_for('process', algorithm=algorithm, user_id=user_id))   
    return render_template('select_user.html', algorithm=algorithm)

@main.route('/metrics')
def metrics():
    eng = Engine("METRICS")
    return render_template('metrics.html')

@main.route('/process/<algorithm>/<user_id>', methods=['GET','POST'])
def process(algorithm, user_id):
    if request.method == 'POST':
        eng = Engine(algorithm)
        recommendations = get_recommendations(eng.preds, user_id)
        recommendations = ', '.join(recommendations)
        return redirect(url_for('recommendations', algorithm=algorithm, user_id=user_id, recommendations=recommendations))
    return render_template('process.html', algorithm=algorithm, user_id=user_id)

@main.route('/process_2/<algorithm>', methods=['GET','POST'])
def process_2(algorithm):
    if request.method == 'POST':
        eng = engines[-1]
        recommendations = get_recommendations(eng.preds, 800)
        recommendations = ', '.join(recommendations)
        return redirect(url_for('recommendations', algorithm=algorithm, recommendations=recommendations))

    return render_template('process_2.html', algorithm=algorithm)

@main.route('/recommendations/<algorithm>/<user_id>/<recommendations>', methods=['GET'])
def recommendations(algorithm, user_id, recommendations):
    return render_template('recommendations.html', algorithm=algorithm, user_id=user_id, recommendations=recommendations)

@main.route('/rate_movie/<algorithm>', methods=['GET', 'POST'])
def rate_movie(algorithm):
    if request.method == 'POST':
        # Get the movie ID and rating from the form submission
        movie_id = request.form['movie_id']
        rating = request.form['rating']
        
        # Convert the rating to a float and add it to the ratings Series
        if rating:
            temp_list=[800,movie_id,float(rating)]
            movie_list.append(temp_list)
        
        # If the user has rated 20 movies, show a "thank you" message
        if len(movie_list) == 20:
            movie_df = pd.DataFrame(movie_list,columns=[userId,movieId,rating])
            eng.run_new_user(movie_df)
            return redirect(url_for('process_2', algorithm=algorithm))
    
    eng = engines[-1]  # get the last engine object added to the list

    title = "NOT IN DATABASE"
    while (title=="NOT IN DATABASE"):
        rando = random.randint(0, len(eng.dp.uniq_movs) - 1)
        tmdb = eng.dp.moviedId_tmdbId_map(rando)
        title = eng.dp.fetch_title(tmdb)        
    
    return render_template('rate_movie.html', title=title, movie_id=rando, eng_id=eng_id)

def get_recommendations(preds, user_id):
    dp = DataProcessor()
    n = 10
    user_id = int(user_id)
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in preds:
        if uid == user_id:
            top_n[iid].append(est)
    # Sort the predictions for each user and retrieve the k highest ones.
    top_rated = dict(sorted(top_n.items(), key=lambda x: x[1], reverse=True)[:n])
    titles = []
    for mId, ratings in top_rated.items():
        titles.append(dp.fetch_title(dp.moviedId_tmdbId_map(mId)))
    return titles

if __name__ == '__main__':
    main.run(debug=True)
