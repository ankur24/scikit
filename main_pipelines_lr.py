import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
import sklearn
import json
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
#function which train model and write info into file

f = open("yelp_review.json", "r")
stars=[]
entries=[]
i=0
for line in f:
    #if i==100: break
    data = json.loads(line)
    stars.append(data['stars'])
   #del data['stars']
    entries.append(data["text"])
    i+=1


#a_train, a_test, b_train, b_test = train_test_split(entries, stars, test_size=0.25, random_state=142)


pipe = make_pipeline(CountVectorizer(), linear_model.LinearRegression())
params = dict(countvectorizer__min_df=[0.005,0.010], countvectorizer__max_df=[0.80,0.90,0.95,1.0])
grid_search = GridSearchCV(pipe, param_grid=params, n_jobs=6)
grid_search.fit(entries, stars)
print("best_params:",grid_search.best_params_)
print("grid_scores:", grid_search.grid_scores_)
print("best_score:", grid_search.best_score_)
joblib.dump(grid_search.best_estimator_, "unipipe_lr_nosw.pkl")



pipe = make_pipeline(TfidfVectorizer(), linear_model.LinearRegression())
params = dict(tfidfvectorizer__min_df=[0.005,0.010, 0.015], tfidfvectorizer__max_df=[0.8, 0.90, 0.95,1.0])
grid_search = GridSearchCV(pipe, param_grid=params, n_jobs=6)
grid_search.fit(entries, stars)
print("best_params:",grid_search.best_params_)
print("grid_scores:", grid_search.grid_scores_)
print("best_score:", grid_search.best_score_)
joblib.dump(grid_search.best_estimator_, "tfidfpipe_lr_nsw.pkl")


class TlinReg(linear_model.LinearRegression):

    def __init__(self, min_df=0.005,max_df=0.8):
        self.min_df = min_df
        self.max_df = max_df
        return super(TlinReg, self).__init__()

    def fit(self, X, y, min_df=0.005,max_df=0.8, *args, **kwargs):
        # Train the model using the training sets
        vect =  CountVectorizer(min_df=self.min_df, max_df=self.max_df, max_features=4500, ngram_range=(2,2))
        vect.fit(X)
        self.bivect  = CountVectorizer(ngram_range=(2,2), vocabulary=vect.vocabulary_)
        super(TlinReg, self).fit(vect.transform(X), y, *args, **kwargs)
        return self

    def predict(self,X, *args, **kwargs):
        return super(TlinReg, self).predict(self.bivect.transform(X))

bimodel = TlinReg()
params = dict(min_df=[0.005,0.010,0.015], max_df=[0.8,0.9,0.95,1.0])
grid_search = GridSearchCV(bimodel, param_grid=params, n_jobs=6)
grid_search.fit(entries, stars)
print("best_params:",grid_search.best_params_)
print("grid_scores:", grid_search.grid_scores_)
print("best_score:", grid_search.best_score_)
joblib.dump(grid_search.best_estimator_, "bipipe_lr_nosw.pkl")
