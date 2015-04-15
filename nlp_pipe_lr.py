"""
## Overview

Unstructured data makes up the vast majority of data.  This is a basic intro to handling unstructured data.  Our objective is to be able to extract the sentiment (positive or negative) from review text.  We will do this from Yelp review data.

Your model will be assessed based on how root mean squared error of the number of stars you predict.  There is a reference solution (which should not be too hard to beat).  The reference solution has a score of 1.

**Download the data here **: http://thedataincubator.s3.amazonaws.com/coursedata/mldata/yelp_train_academic_dataset_review.json.gz


## Download and parse the data

The data is in the same format as in ml.py

"""
from sklearn.externals import joblib
from lib import QuestionList, Question, list_or_dict, ListValidateMixin, YelpListOrDictValidateMixin
QuestionList.set_name("nlp")


class NLPValidateMixin(YelpListOrDictValidateMixin, Question):

  def __init__(self):
      self.regrmodel = None
      self.vecmodel = None
      self.pipemodel = None
      if self.regrfilename: self.regrmodel = joblib.load("questions/"+self.regrfilename)
      if self.vecfilename: self.vecmodel = joblib.load("questions/"+self.vecfilename)
      if self.pipefilename: self.pipemodel = joblib.load("questions/"+self.pipefilename)

  @classmethod
  def fields(cls):
    return ['text']

  @classmethod
  def _test_json(cls):
    return [
      {"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": "WsGQfLLy3YlP_S9jBE3j1w", "review_id": "kzFlI35hkmYA_vPSsMcNoQ", "stars": 5, "date": "2012-11-03", "text": "Love it!!!!! Love it!!!!!! love it!!!!!!!   Who doesn't love Culver's!", "type": "review", "business_id": "LRKJF43s9-3jG9Lgx4zODg"},
      {"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": "Veue6umxTpA3o1eEydowZg", "review_id": "Tfn4EfjyWInS-4ZtGAFNNw", "stars": 3, "date": "2013-12-30", "text": "Everything was great except for the burgers they are greasy and very charred compared to other stores.", "type": "review", "business_id": "LRKJF43s9-3jG9Lgx4zODg"},
      {"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": "u5xcw6LCnnMhddoxkRIgUA", "review_id": "ZYaS2P5EmK9DANxGTV48Tw", "stars": 5, "date": "2010-12-04", "text": "I really like both Chinese restaurants in town.  This one has outstanding crab rangoon.  Love the chicken with snow peas and mushrooms and General Tso Chicken.  Food is always ready in 10 minutes which is accurate.  Good place and they give you free pop.", "type": "review", "business_id": "RgDg-k9S5YD_BaxMckifkg"},
      {"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": "kj18hvJRPLepZPNL7ySKpg", "review_id": "uOLM0vvnFdp468ofLnszTA", "stars": 3, "date": "2011-06-02", "text": "Above average takeout with friendly staff. The sauce on the pan fried noodle is tasty. Dumplings are quite good.", "type": "review", "business_id": "RgDg-k9S5YD_BaxMckifkg"},
      {"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": "L5kqM35IZggaPTpQJqcgwg", "review_id": "b3u1RHmZTNRc0thlFmj2oQ", "stars": 4, "date": "2012-05-28", "text": "We order from Chang Jiang often and have never been disappointed.  The menu is huge, and can accomodate anyone's taste buds.  The service is quick, usually ready in 10 minutes.", "type": "review", "business_id": "RgDg-k9S5YD_BaxMckifkg"}
    ]


@QuestionList.add
class BagOfWordsModel(NLPValidateMixin):
  """
  Build a bag of words model.  Our strategy will be to build a linear model based on the count of the words in each document (review).  **Note:** `def solution` takes an argument `record`.  Samples of `record` are given in `_test_json`.

  1. Don't forget to use tokenization!  This is important for good performance but it is also the most expensive step.  Try vectorizing as a first initial step:
    ``` python
    X = (feature_extraction.text
            .CountVectorizer()
            .fit_transform(text))
    y = scores
    ```
    and then running grid-serach and cross-validation only on of this pre-processed data.

    `CountVectorizer` has to memorize the mapping between words and the index to which it is assigned.  This is linear in the size of the focabulary.  The `HashingVectorizer` does not have to remember this mapping and will lead to much smaller models.

  2. Try choosing different values for `min_df` (minimum document frequency cutoff) and `max_df` in `CountVectorizer`.  Setting `min_df` to zero admits rare words which might only appear once in the entire corpus.  This is both prone to overfitting and makes your data unmanageablely large.  Don't forget to use cross-validation or to select the right value.

  3. Try using `LinearRegression` or `RidgeCV`.  If the memory footprint is too big, try switching to Stochastic Gradient Descent: [`SGDRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html).  You might find that even ordinary linear regression fails due to the data size.  Don't forget to use `GridSearchCV` to determine the regularization parameter!  How do the regularization parameter `alpha` and the values of `min_df` and `max_df` from `CountVectorizer` change the answer?
  """

  regrfilename = ""
  vecfilename = ""
  pipefilename = "unipipe_lr.pkl"
  @list_or_dict
  def solution(self, review):
    #vecdata = self.vecmodel.transform([review["text"]])
    #pred=self.regrmodel.predict(vecdata)
    pred=self.pipemodel.predict([review["text"]])
    return int(round(pred[0]))


@QuestionList.add
class NormalizedModel(NLPValidateMixin):
  """
  Normalization is a key for linear regression.  Previously, we used the count as the normalization scheme.  Try some of these alternative vectorizations:

  1. You can use the "does this word present in this document" as a normalization scheme, which means the values are always 1 or 0.  So we give no additional weight to the presence of the word multiple times.

  2. Try using the log of the number of counts (or more precisely, $log(x+1)$).  This is often used because we want the repeated presence of a word to count for more but not have that effect tapper off.

  3. [TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a common normalization scheme used in text processing.  Use the `TFIDFTransformer`.  There are options for using `idf` and taking the logarithm of `tf`.  Do these significantly affect the result?

  Finally, if you can't decide which one is better, don't forget that you can combine models with a linear regression.
  """
  regrfilename = ""
  vecfilename = ""
  pipefilename = "tfidfpipe_lr.pkl"
  @list_or_dict
  def solution(self, review):
    #vecdata = self.vecmodel.transform([review["text"]])
    #pred=self.regrmodel.predict(vecdata)
    pred=self.pipemodel.predict([review["text"]])
    return int(round(pred[0]))


@QuestionList.add
class BigramModel(NLPValidateMixin):
  """
  In a bigram model, we don't just consider word counts, but also all pairs of consecutive words that appear.  This is going to be a much higher dimensional problem (large $p$) so you should be careful about overfitting.

  Sometimes, reducing the dimension can be useful.  Because we are dealing with a sparse matrix, we have to use `TruncatedSVD`.  If we reduce the dimensions, we can use a more sophisticated models than linear ones.
  """
  regrfilename = ""
  vecfilename = ""
  pipefilename = "bipipe_lr.pkl"
  @list_or_dict
  def solution(self, review):
    #vecdata = self.vecmodel.transform([review["text"]])
    #pred=self.regrmodel.predict(vecdata)
    pred=self.pipemodel.predict([review["text"])
    return int(round(pred[0]))


@QuestionList.add
class FoodBigrams(ListValidateMixin, Question):
  """
  Look over all reviews of restaurants (you may need to look at the dataset from `ml.py` to figure out which ones correspond to restaurants).  There are many bigrams, but let's look at bigrams that are 'special'.  We can think of the corpus as defining an empirical distribution over all ngrams.  We can find word pairs that are unlikely to occur consecutively based on the underlying probability of their words.  Mathematically, if $p(w)$ be the probability of a word $w$ and $p(w_1 w_2)$ is the probability of the bigram $w_1 w_2$, then we want to look at word pairs $w_1 w_2$ where the statistic

  $$ p(w_1 w_2) / p(w_1) / p(w_2) $$

  is high.  Return the top 100 (mostly food) bigrams with this statistic with the 'right' prior factor (see below).

  **Questions:** (to think about: they are not a part of the answer).  This statistic is a ratio and problematic when the denominator is small.  We can fix this by applying Bayesian smoothing to $p(w)$ (i.e. mixing the empirical distribution with the uniform distribution over the vocabulary).

    1. How does changing this smoothing parameter effect the word paris you get qualitatively?

    2. We can interpret the smoothing parameter as adding a constant number of occurences of each word to our distribution.  Does this help you determine set a reasonable value for this 'prior factor'?

    3. Note that this is similar to [Amazon's Statistically Improbable Phrases](http://en.wikipedia.org/wiki/Statistically_Improbable_Phrases).
  """
  def solution(self):
    return [u'huevos rancheros'] * 100
