import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import nltk
import string
from nltk import word_tokenize
from nltk.util import bigrams, trigrams
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import json
from collections import OrderedDict

def count_number_of_positive_words(document):
    return len(list((filter(lambda tok : tok in POSITIVE_WORDS, document))))
def count_number_of_negative_words(document):
    return len(list((filter(lambda tok : tok in NEGATIVE_WORDS, document))))
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

turbo_csv_filename = os.path.join("./", 'final.csv')
TURBO_DF = pd.read_csv(turbo_csv_filename)
TURBO_DF.head()
final_DF = TURBO_DF.iloc[0:5000, :]
POSITIVE_WORDS = set([line.strip() for line in open('positive-words.txt', 'r')])
NEGATIVE_WORDS = set([line.strip() for line in open('negative-words.txt', 'r')])
NLTK_STOPWORDS = set(stopwords.words('english'))
MORE_STOPWORDS = set([line.strip() for line in open('more_stopwords.txt', 'r')])
final_DF['text'] = final_DF['text'].apply(textProcess)
final_DF['positive_words_count'] = final_DF.text.apply(count_number_of_positive_words)
final_DF['negative_words_count'] = final_DF.text.apply(count_number_of_negative_words)
#final_DF['neutral_words_count'] = final_DF.review_length - (df_with_refeature_engineered.positive_words_count + df_with_refeature_engineered.negative_words_count)
final_DF['all_sentiment_words_count'] = final_DF['positive_words_count'] + final_DF['negative_words_count']
final_DF['positive'] = (final_DF['positive_words_count'] * 1.0)/ (final_DF['all_sentiment_words_count'] * 1.0)
final_DF['positive'] = final_DF['positive'] = final_DF['positive'].fillna(2)
final_DF = final_DF.drop(final_DF[final_DF.positive ==2].index)

TEST_SIZE = 0.40
train_review, test_review, train_stars, test_stars = train_test_split(final_DF[['user_average_stars','positive']],final_DF['review_stars'],test_size=TEST_SIZE,random_state=42)
forest100 = RandomForestClassifier(n_estimators = 100, random_state=42)
forest100.fit(train_review, train_stars)
forest100_pred = forest100.predict(test_review)

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

random_forest_confusion_matrix = confusion_matrix(test_stars, forest100_pred)
plot_confusion_matrix(random_forest_confusion_matrix, 'Random Forest (100 Learners) Confusion Matrix',savefilename='RandomForestCM.png')	
	
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

print_classifier_performance_metrics('Random Forest (100 Learners)', forest100_pred)	