from flask import Flask, render_template, request, redirect, url_for
from collections import defaultdict
from engine import Engine
from DataProcessor import DataProcessor
import random

import pandas as pd

main = Flask(__name__)

movie_list = []
engines = []

#Root page
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

#Receives algorithm from index page allows slection os user or yourself
@main.route('/select-user/<algorithm>', methods=['GET', 'POST'])
def select_user(algorithm):
    if request.method == 'POST':
        user_id = request.form['user_id']
        if 'rate_self' in request.form:
            eng = Engine(algorithm)
            engines.append(eng)
            return redirect(url_for('rate_movie', algorithm=algorithm))
        return redirect(url_for('process', algorithm=algorithm, user_id=user_id))   
    return render_template('select_user.html', algorithm=algorithm)

#Renders metrics dataframe to html page
@main.route('/metrics')
def metrics():
    eng = Engine("METRICS")
    results = eng.performance_df.to_html()
    return render_template('metrics.html', scores=results)

#Intermittent page while the engine is creating the predictions for selected algorithm
#after selecting user id and before redirecting to recommendations.html
@main.route('/process/<algorithm>/<user_id>', methods=['GET','POST'])
def process(algorithm, user_id):
    if request.method == 'POST':
        eng = Engine(algorithm)
        recommendations = get_recommendations(eng.preds, user_id)
        recommendations = ', '.join(recommendations)
        return redirect(url_for('recommendations', algorithm=algorithm, user_id=user_id, recommendations=recommendations))
    return render_template('process.html', algorithm=algorithm, user_id=user_id)

#Intermittent page in between self ratings movies and waiting for recommendations to render on recommendations.html
@main.route('/process_2/<algorithm>', methods=['GET','POST'])
def process_2(algorithm):
    if request.method == 'POST':
        eng = engines[-1]
        recommendations = get_recommendations(eng.preds, 800)
        recommendations = ', '.join(recommendations)
        return redirect(url_for('recommendations', algorithm=algorithm, user_id=800, recommendations=recommendations))

    return render_template('process_2.html', algorithm=algorithm)

#page that displays recommendations
@main.route('/recommendations/<algorithm>/<user_id>/<recommendations>', methods=['GET'])
def recommendations(algorithm, user_id, recommendations):
    return render_template('recommendations.html', algorithm=algorithm, user_id=user_id, recommendations=recommendations)

#function and pages that prompt a user to rate a randomly generated title until the user has rated 20 in order
#to create a sufficent preliminary preference profile in order for the algorithm to generate meaningful recommendations
@main.route('/rate_movie/<algorithm>', methods=['GET', 'POST'])
def rate_movie(algorithm):
    print(engines)
    eng = engines[-1]  #retrieves the latest engine object

    if request.method == 'POST':
        # Get the movie ID and rating from the form submission
        movie_id = request.form['movie_id']
        rating = request.form['rating']
        
        # If user picekd a non-skip value
        if rating:
            temp_list=[800,movie_id,float(rating)]
            movie_list.append(temp_list)
        
        #Don't break out of prompting and collecting ratings until user has reached 20 non-skip values
        if len(movie_list) == 20:
            movie_df = pd.DataFrame(movie_list,columns=["userId","movieId","rating"])
            eng.run_new_user(movie_df)
            engines.append(eng)
            return redirect(url_for('process_2', algorithm=algorithm))
    
    title = "NOT IN DATABASE"
    while (title=="NOT IN DATABASE"):
        rando = random.choice(eng.dp.uniq_movs)
        tmdb = eng.dp.moviedId_tmdbId_map(rando)
        title = eng.dp.fetch_title(tmdb)        
    
    return render_template('rate_movie.html', title=title, movie_id=rando)

# takes in a engine.predictions type which are the results of fitting the trainset.antitestset
# and then returns the top 10 predictions for the specified user.  engine.predictions is only the set of 
# unseen movies due to it being composed from the antitestset which are all the unobserved user-movie interactions 
def get_recommendations(preds, user_id):
    dp = DataProcessor()
    user_id = int(user_id)
    #a dictionary of item:rating pair 
    top_n = defaultdict(list)
    #preds is a list of tuples, we only want the specified user and the estimated(predicted) rating (est)
    for uid, iid, true_r, est, _ in preds:
        if uid == user_id:
            top_n[iid].append(est)
    # Sort the predictions for the user and return the top 10.
    top_rated = dict(sorted(top_n.items(), key=lambda x: x[1], reverse=True)[:10])
    titles = []
    for mId, ratings in top_rated.items():
        #need to map the movieId as seen in ratings matrix to proper 
        #tmdbId in order to fetch the title from the movies metadata file
        #all invalid or missing tmdbIds and their associated movies should 
        #be filtered upon initialization of the DataProcessor object.
        titles.append(dp.fetch_title(dp.moviedId_tmdbId_map(mId)))
    return titles

if __name__ == '__main__':
    main.run(debug=True)
