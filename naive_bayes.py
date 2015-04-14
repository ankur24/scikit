import numpy as np
from numpy.random import rand
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import json

f = open("yelp_review.json", "r")
stars=[]
entries=[]
i=0
for line in f:
    #if i==100: break
    data = json.loads(line)
    s = data['stars']
    if s == 2:
        s = 1
    elif s == 4:
        s = 5
    stars.append(s)
    del data['stars']
    entries.append(data)
    i+=1
    
a_train, a_test, b_train, b_test = train_test_split(entries, stars, test_size=0.20, random_state=14)

vect = CountVectorizer(stop_words='english',  ngram_range=(2,2))
vect.fit(e['text'] for e in entries)
dvect = CountVectorizer(ngram_range=(2,2), vocabulary=vect.vocabulary_)
joblib.dump(dvect, "nbmodels/bivect.pkl")

#joblib.dump(vect, "bigram_hashing_vect.pkl")

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(vect.transform(e['text'] for e in a_train), b_train)
joblib.dump(clf, "nbmodels/naive_bayes_bigram.pkl")
vect = joblib.load("bivect.pkl")
clf = joblib.load("naive_bayes_bigram.pkl")
print clf.score(vect.transform(e['text'] for e in a_test), b_test)
