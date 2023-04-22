from flask import Flask, render_template, request, redirect, url_for
from collections import defaultdict
from engine import Engine
from DataProcessor import DataProcessor

main = Flask(__name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        algorithm = request.form['algorithm']
        return redirect(url_for('select_user', algorithm=algorithm))
    return render_template('index.html')

@main.route('/select-user/<algorithm>', methods=['GET', 'POST'])
def select_user(algorithm):
    if request.method == 'POST':
        user_id = request.form['user_id']
        alg = 1 if algorithm == "SVD" else 2
        eng = Engine(alg)
        preds = eng.run()
        recommendations = get_recommendations(preds, algorithm, user_id)
        recommendations = ', '.join(recommendations)
        return redirect(url_for('recommendations', algorithm=algorithm, user_id=user_id, recommendations=recommendations))
   
    return render_template('select_user.html', algorithm=algorithm)

@main.route('/recommendations/<algorithm>/<user_id>/<recommendations>', methods=['GET'])
def recommendations(algorithm, user_id, recommendations):
    return render_template('recommendations.html', algorithm=algorithm, user_id=user_id, recommendations=recommendations)

def get_recommendations(preds, algorithm, user_id):
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
