import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

pd.set_option('display.max_columns', 36)
print(pd.__version__)

import nltk
import string
from nltk import word_tokenize
from nltk.util import bigrams, trigrams
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter

turbo_csv_filename = os.path.join("./", 'final.csv')
TURBO_DF = pd.read_csv(turbo_csv_filename)
TURBO_DF.head()
TURBO_DF = TURBO_DF[TURBO_DF['user_review_count'] >= 50]
TURBO_DF = TURBO_DF[TURBO_DF['business_review_count'] >= 50]

business_average_star = TURBO_DF.groupby(['business_id']).mean()
star_category_absolute_frequencies2 = (business_average_star.business_average_stars).round().value_counts()
good_business = TURBO_DF[TURBO_DF['business_average_stars'] >= 4.10]
average_business = TURBO_DF[TURBO_DF['business_average_stars'] >= 3.2]
average_business =average_business[average_business['business_average_stars'] <= 3.5]
bad_business = TURBO_DF[TURBO_DF['business_average_stars'] <= 2.5]

user_average_star = TURBO_DF.groupby(['user_id']).mean()
star_category_absolute_frequencies3 = (user_average_star.user_average_stars).round().value_counts(ascending=True)
good_user = TURBO_DF[TURBO_DF['user_average_stars'] >= 4.0]
#good_user = TURBO_DF[TURBO_DF['user_average_stars'] <= 4.8]
average_user = TURBO_DF[TURBO_DF['user_average_stars'] >= 3.1]
average_user =average_user[average_user['user_average_stars'] <= 3.5]
bad_user = TURBO_DF[TURBO_DF['user_average_stars'] <= 3]
bad_user = bad_user[bad_user['user_average_stars'] >= 0]



def tableJoin(user, good, average, bad):
    cols_to_use = ['user_id','business_id','date']
    goodgood = pd.merge(user, good[cols_to_use], how='inner')
    goodaverage = pd.merge(user, average[cols_to_use], how='inner')
    goodbad = pd.merge(user, bad[cols_to_use], how='inner')
    return goodgood, goodaverage, goodbad

goodbusiness, averbusiness, badbusiness = tableJoin(good_user, good_business, average_business, bad_business)
avergood, averaver, averbad = tableJoin(average_user, good_business, average_business, bad_business)
badgood, badaver, badbad = tableJoin(bad_user, good_business, average_business, bad_business)
goodgood, goodaver, goodbad = tableJoin(good_user, good_business, average_business, bad_business)

def Concat(good, aver, bad):
    frames = [good, aver, bad]
    together = pd.concat(frames)
    return together

user_good = Concat(goodgood, goodaver, goodbad)
user_aver = Concat(avergood, averaver, averbad)
user_bad = Concat(badgood, badaver, badbad)


def ChangeAverage(data, value):
    data['BP_average_stars'] = value
    return data

user_good = ChangeAverage(user_good, 4.357)
user_aver = ChangeAverage(user_aver, 3.27)
user_bad = ChangeAverage(user_bad,1.699)

user_all = Concat(user_good,user_aver,user_bad)
frames = [user_good,user_bad]
#frames = [user_good, user_bad]
#user_all = pd.concat(frames)

POSITIVE_WORDS = set([line.strip() for line in open('positive-words.txt', 'r')])
NEGATIVE_WORDS = set([line.strip() for line in open('negative-words.txt', 'r')])
NLTK_STOPWORDS = set(stopwords.words('english'))
MORE_STOPWORDS = set([line.strip() for line in open('more_stopwords.txt', 'r')])

def textProcess(s):
    s = s.lower()
    #print(s)
    s = s.translate(str.maketrans(dict.fromkeys(string.punctuation, None)))
    # may consider removing arabic-hindu digits
    token_list = nltk.word_tokenize(s)
    #print(token_list)
    #return token_list
    exclude_stopwords = lambda token : token not in NLTK_STOPWORDS
    return list(filter(exclude_stopwords, token_list))

import json
from collections import OrderedDict

def count_number_of_positive_words(document):
    return len(list((filter(lambda tok : tok in POSITIVE_WORDS, document))))

def count_number_of_negative_words(document):
    return len(list((filter(lambda tok : tok in NEGATIVE_WORDS, document))))


user_all['text'] = user_all['text'].apply(textProcess)
user_all['positive_words_count'] = user_all.text.apply(count_number_of_positive_words)
user_all['negative_words_count'] = user_all.text.apply(count_number_of_negative_words)
#final_DF['neutral_words_count'] = final_DF.review_length - (df_with_refeature_engineered.positive_words_count + df_with_refeature_engineered.negative_words_count)
user_all['all_sentiment_words_count'] = user_all['positive_words_count'] + user_all['negative_words_count']

user_all['positive'] = (user_all['positive_words_count'] * 1.0)/ (user_all['all_sentiment_words_count'] * 1.0)
user_all['negative'] = (user_all['negative_words_count'] * 1.0)/ (user_all['all_sentiment_words_count'] * 1.0)
user_all['ratio'] = (user_all['positive_words_count'] * 1.0)/ (user_all['negative_words_count'] * 1.0)
user_all['positive'] = user_all['positive'] = user_all['positive'].fillna(2)
user_all = user_all.drop(user_all[user_all.positive ==2].index)
user_all['average_diff'] = (user_all['BP_average_stars'] * 1.0) - (user_all['user_average_stars'] * 1.0)
user_all['average_diff_ratio'] = (user_all['average_diff'] * 1.0) / 5
user_all = user_all.replace('inf', 999)
user_all

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score
TEST_SIZE = 0.20


#train_review, test_review, train_stars, test_stars = train_test_split(user_all[['BP_average_stars','positive','average_diff'
#                        'negative','ratio','positive_words_count','negative_words_count','all_sentiment_words_count']],
#                         user_all['review_stars'],test_size=TEST_SIZE,random_state=42)
train_review, test_review, train_stars, test_stars = train_test_split(user_all[['positive','BP_average_stars','negative','ratio']],user_all['review_stars'], test_size=TEST_SIZE, random_state=42)
						 
def make_confusion_matrix_relative(confusion_matrix):
    star_category_classes = [1, 2, 3, 4, 5]
    N = list(map(lambda clazz : sum(test_stars == clazz), star_category_classes))
    relative_confusion_matrix = np.empty((len(star_category_classes), len(star_category_classes)))
    
    for j in range(0, len(star_category_classes)):
        if N[j] > 0:
            relative_frequency = confusion_matrix[j, :] / float(N[j])
            relative_confusion_matrix[j, :] = relative_frequency
            
    return relative_confusion_matrix

# http://www.wenda.io/questions/4330313/heatmap-with-text-in-each-cell-with-matplotlibs-pyplot.html
# http://stackoverflow.com/questions/20520246/create-heatmap-using-pandas-timeseries
# http://sebastianraschka.com/Articles/heatmaps_in_r.html
# http://code.activestate.com/recipes/578175-hierarchical-clustering-heatmap-python/
def plot_confusion_matrix(confusion_matrix=[[]], title='CM', savefilename=''):
    rcm = make_confusion_matrix_relative(confusion_matrix)
    #plt.imshow(rcm, vmin=0, vmax=1, interpolation='nearest')
    c = plt.pcolor(rcm, edgecolors='k', linewidths=4, cmap='jet', vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(0.5 + np.arange(5), np.arange(1,6))
    plt.yticks(0.5 + np.arange(5), np.arange(1,6))

    def show_values(pc, fmt="%.2f", **kw):
        pc.update_scalarmappable()
        ax = pc.get_axes()
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if sum(color[:2] > 0.3) >= 2:
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)
    
    show_values(c)

    if savefilename:
        plt.savefig(savefilename, bbox_inches='tight')
    
    return plt.show()
	
def print_classifier_performance_metrics(name, predictions):
    target_names = ['1 star', '2 star', '3 star', '4 star', '5 star']
    
    print("MODEL: %s" % name)
    print()

    print('Precision: ' + str(metrics.precision_score(test_stars, predictions))) 
    print('Recall: ' + str(metrics.recall_score(test_stars, predictions))) 
    print('F1: ' + str(metrics.f1_score(test_stars, predictions)))
    print('Accuracy: ' + str(metrics.accuracy_score(test_stars, predictions)))

    print()
    print('Classification Report:') 
    print(classification_report(test_stars, predictions, target_names=target_names)) 
    
    print()
    print('Precision variance: %f' % np.var(precision_score(test_stars, predictions, average=None), ddof=len(target_names)-1)) 
    
    print()
    print('Recall variance: %f' % np.var(recall_score(test_stars, predictions, average=None), ddof=len(target_names)-1)) 
	
	
forest100 = RandomForestClassifier(n_estimators = 100, random_state=42)
forest100.fit(train_review, train_stars)
forest100_pred = forest100.predict(test_review)
random_forest_confusion_matrix = confusion_matrix(test_stars, forest100_pred)
plot_confusion_matrix(random_forest_confusion_matrix, 'Random Forest (100 Learners) Confusion Matrix',savefilename='RandomForestCM.png')
print_classifier_performance_metrics('Random Forest (100 Learners)', forest100_pred)

probabilities = forest100.predict_proba(test_review)
print(probabilities)
print(forest100_pred)

'''
multinomial_nb_classifier = MultinomialNB()
multinomial_nb_classifier.fit(train_review, train_stars)
multinomial_nb_prediction = multinomial_nb_classifier.predict(test_review) 
multinomial_confusion_matrix = confusion_matrix(test_stars, multinomial_nb_prediction)
print(make_confusion_matrix_relative(multinomial_confusion_matrix))
plot_confusion_matrix(multinomial_confusion_matrix,'Multinomial Naive Bayes Confusion Matrix', savefilename='MultinomialCM.png')
print_classifier_performance_metrics('Multinomial Naive Bayes', multinomial_nb_prediction)


svc = SVC()
svc.fit(train_review, train_stars) 
svc_predictions = svc.predict(test_review)
np.save('svcPred', svc_predictions)
if os.path.isfile('svcPred.npy'):
    svc_predictions = np.load('svcPred.npy')
svc_confusion_matrix = confusion_matrix(test_stars, svc_predictions)
plot_confusion_matrix(svc_confusion_matrix, 'SVC Confusion Matrix', savefilename='SVC_CM.png')
print_classifier_performance_metrics('SVC', svc_predictions)


bernoulli_nb_classifier = BernoulliNB().fit(train_review, train_stars)
bernoulli_nb_prediction = bernoulli_nb_classifier.predict(test_review)
bernoulli_confusion_matrix = confusion_matrix(test_stars, bernoulli_nb_prediction)
plot_confusion_matrix(bernoulli_confusion_matrix, 'Bernoulli Naive Bayes Confusion Matrix', savefilename='BernoulliCM.png')
print_classifier_performance_metrics('Bernoulli Naive Bayes', bernoulli_nb_prediction)

'''