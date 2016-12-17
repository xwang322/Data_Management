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

pd.set_option('display.max_columns', 36)
#print(pd.__version__)
POSITIVE_WORDS = set([line.strip() for line in open('positive-words.txt', 'r')])
NEGATIVE_WORDS = set([line.strip() for line in open('negative-words.txt', 'r')])
NLTK_STOPWORDS = set(stopwords.words('english'))
MORE_STOPWORDS = set([line.strip() for line in open('more_stopwords.txt', 'r')])

def remove_numbers_in_string(s):
    return s.translate(str.maketrans(dict.fromkeys(string.digits, None)))

def lowercase_remove_punctuation(s):
    s = s.lower()
    return s.translate(str.maketrans(dict.fromkeys(string.punctuation, None))) 
    
def remove_stopwords(s):
    token_list = nltk.word_tokenize(s)
    exclude_stopwords = lambda token : token not in NLTK_STOPWORDS
    return ' '.join(filter(exclude_stopwords, token_list))

def filter_out_more_stopwords(token_list):
    return filter(lambda tok : tok not in MORE_STOPWORDS, token_list)

def stem_token_list(token_list):
    STEMMER = PorterStemmer()
    return [STEMMER.stem(tok) for tok in token_list]

def restring_tokens(token_list):
    return ' '.join(token_list)

def lowercase_remove_punctuation_and_numbers_and_tokenize_and_filter_more_stopwords_and_stem_and_restring(s):
    s = remove_numbers_in_string(s)
    s = lowercase_remove_punctuation(s)
    s = remove_stopwords(s)
    token_list = nltk.word_tokenize(s)
    token_list = filter_out_more_stopwords(token_list)
    token_list = stem_token_list(token_list)
    return restring_tokens(token_list)

turbo_csv_filename = os.path.join("./", 'final.csv')
TURBO_DF = pd.read_csv(turbo_csv_filename)
TURBO_DF.head()
'''
for idx in range(5):
    print(TURBO_DF.review_text[idx])
    print
'''

def randomSample(data, length):
    total_len = len(data)
    frac1 = length / total_len
    #print(frac1)
    sample = data.sample(frac = frac1, replace = True)
    return sample

sample = randomSample(TURBO_DF, 50000)
TURBO_DF = sample

initial_features = ['business_id', 'business_name', 'review_stars', 'text','user_average_stars']
df_with_initial_features_and_preprocessed_review_text = TURBO_DF[initial_features]
df_with_initial_features_and_preprocessed_review_text['text'] = df_with_initial_features_and_preprocessed_review_text['text'].apply(lowercase_remove_punctuation_and_numbers_and_tokenize_and_filter_more_stopwords_and_stem_and_restring)
'''
for idx in range(5):
    print(df_with_initial_features_and_preprocessed_review_text.review_text[idx])
    print
'''
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score
from scipy import sparse

TEST_SIZE = 0.40
# this average score needs to be changed
train_review, test_review, train_score, test_score, train_aver, test_aver = train_test_split(df_with_initial_features_and_preprocessed_review_text.text, df_with_initial_features_and_preprocessed_review_text.review_stars, df_with_initial_features_and_preprocessed_review_text.user_average_stars, test_size=TEST_SIZE, random_state=42)

bag_of_words_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,ngram_range = (1, 1),binary = False,strip_accents='unicode')
binary_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,ngram_range = (1, 1),binary = True,strip_accents='unicode')
bigram_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,ngram_range = (2, 2),strip_accents='unicode')
trigram_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,ngram_range = (3, 3),strip_accents='unicode')
bi_and_trigram_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,ngram_range = (2,3),strip_accents='unicode')
random_forest_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,ngram_range = (1,1),strip_accents = 'unicode',max_features = 1000)
new_vectorizer = CountVectorizer(ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)

feature_matrix_train_review = bi_and_trigram_vectorizer.fit_transform(train_review)
feature_matrix_train_averscore = DataFrame.as_matrix(train_aver)
feature_matrix_train_averscore_matrix = (sparse.csr_matrix(feature_matrix_train_averscore)).transpose()
feature_matrix_train = sparse.hstack((feature_matrix_train_averscore_matrix, feature_matrix_train_review))
feature_matrix_test_review = bi_and_trigram_vectorizer.transform(test_review)
feature_matrix_test_averscore = DataFrame.as_matrix(test_aver)
feature_matrix_test_averscore_matrix = (sparse.csr_matrix(feature_matrix_test_averscore)).transpose()
feature_matrix_test = sparse.hstack((feature_matrix_test_averscore_matrix, feature_matrix_test_review))
'''
# multinominal naive_bayes
multinomial_nb_classifier = MultinomialNB()
multinomial_nb_classifier.fit(feature_matrix_train, train_score)
multinomial_nb_prediction = multinomial_nb_classifier.predict(feature_matrix_test)
'''
def make_confusion_matrix_relative(confusion_matrix):
    star_category_classes = [1, 2, 3, 4, 5]
    N = list(map(lambda clazz : sum(test_score == clazz), star_category_classes))
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
    plt.close()
    return True
'''
multinomial_confusion_matrix = confusion_matrix(test_score, multinomial_nb_prediction)
print(make_confusion_matrix_relative(multinomial_confusion_matrix))
plot_confusion_matrix(multinomial_confusion_matrix,'Multinomial Naive Bayes Confusion Matrix', savefilename='MultinomialCM.png')
'''
def print_classifier_performance_metrics(name, predictions):
    target_names = ['1 star', '2 star', '3 star', '4 star', '5 star']
    
    print("MODEL: %s" % name)
    print()

    print('Precision: ' + str(metrics.precision_score(test_score, predictions))) 
    print('Recall: ' + str(metrics.recall_score(test_score, predictions))) 
    print('F1: ' + str(metrics.f1_score(test_score, predictions)))
    print('Accuracy: ' + str(metrics.accuracy_score(test_score, predictions)))

    print()
    print('Classification Report:') 
    print(classification_report(test_score, predictions, target_names=target_names)) 
    
    print()
    print('Precision variance: %f' % np.var(precision_score(test_score, predictions, average=None), ddof=len(target_names)-1)) 
    
    print()
    print('Recall variance: %f' % np.var(recall_score(test_score, predictions, average=None), ddof=len(target_names)-1)) 
'''
print_classifier_performance_metrics('Multinomial Naive Bayes', multinomial_nb_prediction)

#Bernoulli Naive Bayes Model
bernoulli_nb_classifier = BernoulliNB().fit(feature_matrix_train, train_score)
bernoulli_nb_prediction = bernoulli_nb_classifier.predict(feature_matrix_test)
bernoulli_confusion_matrix = confusion_matrix(test_score, bernoulli_nb_prediction)
plot_confusion_matrix(bernoulli_confusion_matrix, 'Bernoulli Naive Bayes Confusion Matrix', savefilename='BernoulliCM.png')
print_classifier_performance_metrics('Bernoulli Naive Bayes', bernoulli_nb_prediction)
'''
# random forest 100 leaners model
forest100 = RandomForestClassifier(n_estimators = 100, random_state=42)
forest100.fit(feature_matrix_train.toarray(), train_score)
forest100_pred = forest100.predict(feature_matrix_test.toarray())
np.save('forest100pred', forest100_pred)
random_forest_confusion_matrix = confusion_matrix(test_score, forest100_pred)
plot_confusion_matrix(random_forest_confusion_matrix, 'Random Forest (100 Learners) Confusion Matrix',savefilename='RandomForestCM.png')
print_classifier_performance_metrics('Random Forest (100 Learners)', forest100_pred)

#SVC model
svc_feature_matrix_train = feature_matrix_train
svc_feature_matrix_test = feature_matrix_test
svc = SVC()
svc.fit(svc_feature_matrix_train, train_score) 
svc_predictions = svc.predict(svc_feature_matrix_test)
np.save('svcPred', svc_predictions)
if os.path.isfile('svcPred.npy'):
    svc_predictions = np.load('svcPred.npy')
svc_confusion_matrix = confusion_matrix(test_score, svc_predictions)
plot_confusion_matrix(svc_confusion_matrix, 'SVC Confusion Matrix', savefilename='SVC_CM.png')
print_classifier_performance_metrics('SVC', svc_predictions)

# First Round Summary
from operator import itemgetter
from functools import reduce
def argmax(dictionary):
    return (max(dictionary.items(), key=itemgetter(1)))[0]

class Multinomial_NB_Classifier():
    
    def train(self, class_labels, documents, class_priors=[], complement=False):       
        Classes = sorted(list(set(class_labels)))
        Vocabulary = reduce(lambda V, d : V.union(set(d.split())), documents, set())
        
        if len(class_priors) != len(Classes):
            N_documents_per_class = lambda c : sum(c == np.array(class_labels))
            N_documents = len(documents)
            class_priors = {c : N_documents_per_class(c) / float(N_documents) for c in Classes}
        
        Text_given_class = {c : documents[c == np.array(class_labels)].sum() for c in Classes}
        length_of_concatenated_documents_of_class = {c : len(Text_given_class[c].split()) for c in Classes}
        total_length_of_concatenated_documents_of_all_classes = sum([length_of_concatenated_documents_of_class[C] for C in Classes])
        
        absolute_frequency_vectorizer = CountVectorizer(analyzer = "word",
                                                        vocabulary = Vocabulary,
                                                        ngram_range = (1, 1),
                                                        binary = False)
        documentclass_termunigram_matrix = absolute_frequency_vectorizer.fit_transform(Text_given_class.values())
    
        number_of_feature_words = documentclass_termunigram_matrix.shape[1]
        feature_word_index = dict(zip(absolute_frequency_vectorizer.get_feature_names(), range(number_of_feature_words)))
        word_counts_irregardless_of_class = documentclass_termunigram_matrix.sum(axis=0)
        
        WORD_GIVEN_CLASS_CPT = {}
        COMPLEMENT_WORD_GIVEN_CLASS_CPT = {}
        K = len(Vocabulary)
        
        for class_idx, c in enumerate(Classes):
            if c not in WORD_GIVEN_CLASS_CPT:
                WORD_GIVEN_CLASS_CPT[c] = {}
                COMPLEMENT_WORD_GIVEN_CLASS_CPT[c] = {}

            for w in Vocabulary:
                T_wc = documentclass_termunigram_matrix[class_idx, feature_word_index[w]]
                WORD_GIVEN_CLASS_CPT[c][w] = float(1 + T_wc) / (K + length_of_concatenated_documents_of_class[c])
                
                C_wc = word_counts_irregardless_of_class[0, feature_word_index[w]] - T_wc
                complement_classes_wc = total_length_of_concatenated_documents_of_all_classes-length_of_concatenated_documents_of_class[c]
                COMPLEMENT_WORD_GIVEN_CLASS_CPT[c][w] = float(1 + C_wc) / (K + complement_classes_wc)
           
        self.classes = Classes
        self.vocabulary = Vocabulary
        self.class_priors = class_priors
        self.words_cpt = WORD_GIVEN_CLASS_CPT
        
        if complement:
            self.complement = True
            self.complement_cpt = COMPLEMENT_WORD_GIVEN_CLASS_CPT
        else:
            self.complement = False
            
        return self

    def predict(self, documents):
        target_labels = []
        
        for d in documents:
            score = {}
            Words_d = set(d.split())
            
            for class_idx, c in enumerate(self.classes):
                word_log_likelihoods = np.log([self.words_cpt[c][w] for w in Words_d if w in self.vocabulary])
                score[c] = np.log(self.class_priors[c])
                
                if self.complement:
                    word_in_complement_classes_log_likelihoods = np.log([self.complement_cpt[c][w] for w in Words_d if w in self.vocabulary])
                    score[c] -= sum(word_in_complement_classes_log_likelihoods)
                else:
                    score[c] += sum(word_log_likelihoods)
                    
            c_map = argmax(score)
            target_labels.append(c_map)
            
        return target_labels 

clf = Multinomial_NB_Classifier().train(class_labels=train_score, documents=train_review, complement=True)
pre = clf.predict(test_review)
my_multinomial_confusion_matrix = confusion_matrix(test_score, pre)
plot_confusion_matrix(my_multinomial_confusion_matrix,'My Multinomial Naive Bayes Confusion Matrix', savefilename='MyMultinomialCM.png')
print_classifier_performance_metrics('My Multinomial Naive Bayes', pre)

# Top 10 Features for each Star Category of New Multinomial NB
N = 10
vocab = np.array([t for t, i in sorted(new_vectorizer.vocabulary_.items(), key=itemgetter(1))])
for i, label in enumerate(sorted(set(train_score))):
    top_n_features_indices = np.argsort(multinomial_nb_classifier.coef_[i])[-N:]
    print("\nThe top %d most informative features for star category %d: \n%s" % (N, label, ", ".join(vocab[top_n_features_indices])))

# jump over the negative and positive part because of review_votes???
'''
# Oversampling Minority Classes
TEST_SIZE = 0.40
train_X, test_X, train_y, test_y = train_test_split(df_with_initial_features_and_preprocessed_review_text.text, df_with_initial_features_and_preprocessed_review_text.review_stars, test_size=TEST_SIZE, random_state=42)

ONE_STAR_LABEL = 1
TWO_STAR_LABEL = 2
THREE_STAR_LABEL = 3
FOUR_STAR_LABEL = 4

one_star_reviews = train_X[train_y == ONE_STAR_LABEL]
two_star_reviews = train_X[train_y == TWO_STAR_LABEL]
three_star_reviews = train_X[train_y == THREE_STAR_LABEL]
four_star_reviews = train_X[train_y == FOUR_STAR_LABEL]

one_star_labels = train_y[train_y == ONE_STAR_LABEL]
two_star_labels = train_y[train_y == TWO_STAR_LABEL]
three_star_labels = train_y[train_y == THREE_STAR_LABEL]

difference_btw_number_of_4_and_1_stars = len(four_star_reviews) - len(one_star_reviews)
difference_btw_number_of_4_and_2_stars = len(four_star_reviews) - len(two_star_reviews)
difference_btw_number_of_4_and_3_stars = len(four_star_reviews) - len(three_star_reviews)

assert (np.array([difference_btw_number_of_4_and_1_stars, difference_btw_number_of_4_and_2_stars, difference_btw_number_of_4_and_3_stars]) > 0).all()
q1 = difference_btw_number_of_4_and_1_stars/len(one_star_reviews)
r1 = difference_btw_number_of_4_and_1_stars - len(one_star_reviews) * q1
q2 = difference_btw_number_of_4_and_2_stars/len(two_star_reviews)
r2 = difference_btw_number_of_4_and_2_stars - len(two_star_reviews) * q2
q3 = difference_btw_number_of_4_and_3_stars/len(three_star_reviews)
r3 = difference_btw_number_of_4_and_3_stars - len(three_star_reviews) * q3
print(q1,q2,q3,r1,r2,r3,len(train_X))

train_X = np.vstack([np.reshape(train_X, (len(train_X), 1)), np.tile(one_star_reviews, (1, q1)).T, np.reshape(one_star_reviews[:r1], (r1, 1)),np.tile(two_star_reviews, (1, q2)).T,np.reshape(two_star_reviews[:r2], (r2, 1)),np.tile(three_star_reviews, (1, q3)).T,np.reshape(three_star_reviews[:r3], (r3, 1))])[:,0]
train_y = np.vstack([np.reshape(train_y, (len(train_y), 1)), np.tile(one_star_labels, (1, q1)).T, np.reshape(one_star_labels[:r1], (r1, 1)), np.tile(two_star_labels, (1, q2)).T, np.reshape(two_star_labels[:r2], (r2, 1)), np.tile(three_star_labels, (1, q3)).T,np.reshape(three_star_labels[:r3], (r3, 1))])[:,0]
clf = Multinomial_NB_Classifier().train(class_labels=train_y, documents=train_X)
oversampling_multinomial_nb_prediction = clf.predict(test_X)
oversampling_multinomial_confusion_matrix = confusion_matrix(test_y, oversampling_multinomial_nb_prediction)
print(make_confusion_matrix_relative(oversampling_multinomial_confusion_matrix))
plot_confusion_matrix(oversampling_multinomial_confusion_matrix,'My Multinomial Naive Bayes Confusion Matrix with Oversampling for Minority Classes', savefilename='MyMultinomialOversamplingCM.png')
print_classifier_performance_metrics('Multinomial Naive Bayes Trained on Oversampling', oversampling_multinomial_nb_prediction)

# Oversampling with Bigram Multinomial Naive Bayes
oversampling_bigram_multinomial_feature_matrix_train = bigram_vectorizer.fit_transform(train_X)
oversampling_bigram_multinomial_feature_matrix_test = bigram_vectorizer.transform(test_X)
oversampling_bigram_multinomial_nb_classifier = MultinomialNB().fit(oversampling_bigram_multinomial_feature_matrix_train, train_y)
oversampling_bigram_multinomial_nb_prediction = oversampling_bigram_multinomial_nb_classifier.predict(oversampling_bigram_multinomial_feature_matrix_test)
oversampling_bigram_multinomial_confusion_matrix = confusion_matrix(test_y, oversampling_bigram_multinomial_nb_prediction)
print(make_confusion_matrix_relative(oversampling_bigram_multinomial_confusion_matrix))
plot_confusion_matrix(oversampling_bigram_multinomial_confusion_matrix,'Bigram Multinomial Naive Bayes with Oversampling \n for Minority Classes Confusion Matrix', savefilename='BigramMultinomialOversamplingCM.png')
print_classifier_performance_metrics('Bigram Naive Bayes Trained on Oversampling', oversampling_bigram_multinomial_nb_prediction)


#Multiclass Logistic Regression Model

TEST_SIZE = 0.40
train_X, test_X, train_y, test_y = train_test_split(df_with_initial_features_and_preprocessed_review_text.text, df_with_initial_features_and_preprocessed_review_text.review_stars, test_size=TEST_SIZE, random_state=42)
BIGRAM_RANGE = (2, 2)
v = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,ngram_range = BIGRAM_RANGE,strip_accents = 'unicode')
logistic_feature_matrix_train = v.fit_transform(train_X)
logistic_feature_matrix_test = v.transform(test_X)
logistic_feature_matrix_train, logistic_feature_matrix_test
'''