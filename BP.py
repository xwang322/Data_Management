from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import node
from graph import Graph

def convertnan(list):
    for each in list:
        if np.isnan(each):
            index = list.index(each)
            list[index] = 1.0
    return list

def generation(distribution):
    list = pd.Series(distribution, index=[1.0, 2.0, 3.0, 4.0, 5.0]).tolist()
    list_convert = convertnan(list)
    #print(list_convert, sum(list_convert))
    list_freq = list_convert/sum(list_convert)
    return list_freq	

def assign(good, aver, tough):
    matrix = np.empty((5,5,5))
    for i in range(5):
        matrix[i,:,:] = tough[i]
    for i in range(5):
        matrix[:,:,i] *= good[i]        
    for i in range(5):
        matrix[:,i,:] *= aver[i]        
    return matrix *100
'''	
def assign(good, tough):
    matrix = np.empty((5,5))
    for i in range(5):
        matrix[i,:] = tough[i]
    for i in range(5):
        matrix[:,i] *= good[i]                
    return matrix *100
'''
def MakeGraph(good, average, bad):

    G = Graph()
    # a is tough user, b is nice user and c is average user
    a = G.addVarNode('a', 5)
    b = G.addVarNode('b', 5)
    c = G.addVarNode('c', 5)
    # p1 is a good restaurant, p2 is a bad restaurant, p3 is average restaurant
	
    p1 = np.array(good)	
    G.addFacNode(p1, a, b, c)

    p2 = np.array(average)	
    G.addFacNode(p2, a, b, c)

    p3 = np.array(bad)	
    G.addFacNode(p3, a, b, c)

    return G

def TestGraph(good, average, bad):

    G = MakeGraph(good, average, bad)
    marg = G.marginals()
    brute = G.bruteForce()

    # check the marginals
    am = marg['a']
    print('a1', am[0])
    print('a2', am[1])
    print('a3', am[2])
    print('a4', am[3])
    print('a5', am[4])

    bm = marg['b']
    print('b1', bm[0])
    print('b2', bm[1])
    print('b3', bm[2])
    print('b4', bm[3])
    print('b5', bm[4])

    cm = marg['c']
    print('c1', cm[0])
    print('c2', cm[1])
    print('c3', cm[2])
    print('c4', cm[3])
    print('c5', cm[4])
	
turbo_csv_filename = os.path.join("./", 'final.csv')
TURBO_DF = pd.read_csv(turbo_csv_filename)
TURBO_DF = TURBO_DF[TURBO_DF['user_review_count'] >= 100]
TURBO_DF = TURBO_DF[TURBO_DF['business_review_count'] >= 100]

business_average_star = TURBO_DF.groupby(['business_id']).mean()
star_category_absolute_frequencies2 = (business_average_star.business_average_stars).round().value_counts()
good_business = TURBO_DF[TURBO_DF['business_average_stars'] >= 4.10]
average_business = TURBO_DF[TURBO_DF['business_average_stars'] >= 3.2]
average_business =average_business[average_business['business_average_stars'] <= 3.5]
bad_business = TURBO_DF[TURBO_DF['business_average_stars'] <= 2.5]
#print(bad_business.head())

user_average_star = TURBO_DF.groupby(['user_id']).mean()
star_category_absolute_frequencies3 = (user_average_star.user_average_stars).round().value_counts(ascending=True)
good_user = TURBO_DF[TURBO_DF['user_average_stars'] >= 4.0]
average_user = TURBO_DF[TURBO_DF['user_average_stars'] >= 3.1]
average_user =average_user[average_user['user_average_stars'] <= 3.5]
bad_user = TURBO_DF[TURBO_DF['user_average_stars'] <= 3.0]
bad_user = bad_user[bad_user['user_average_stars'] >= 0]
#print(bad_user.head())

gooduser_goodbusiness = pd.merge(good_user, good_business, on = ['business_id', 'user_id'], how = 'inner')
gooduser_averbusiness = pd.merge(good_user, average_business, on = ['business_id', 'user_id'], how = 'inner')
gooduser_badbusiness = pd.merge(good_user, bad_business, on = ['business_id', 'user_id'], how = 'inner')
averuser_goodbusiness = pd.merge(average_user, good_business, on = ['business_id', 'user_id'], how = 'inner')
averuser_averbusiness = pd.merge(average_user, average_business, on = ['business_id', 'user_id'], how = 'inner')
averuser_badbusiness = pd.merge(average_user, bad_business, on = ['business_id', 'user_id'], how = 'inner')
baduser_goodbusiness = pd.merge(bad_user, good_business, on = ['business_id', 'user_id'], how = 'inner')
baduser_averbusiness = pd.merge(bad_user, average_business, on = ['business_id', 'user_id'], how = 'inner')
baduser_badbusiness = pd.merge(bad_user, bad_business, on = ['business_id', 'user_id'], how = 'inner')
#print(baduser_badbusiness.head())

distribution1 = (gooduser_goodbusiness.review_stars_x).round().value_counts()
distribution2 = (gooduser_averbusiness.review_stars_x).round().value_counts()
distribution3 = (gooduser_badbusiness.review_stars_x).round().value_counts()
distribution4 = (averuser_goodbusiness.review_stars_x).round().value_counts()
distribution5 = (averuser_averbusiness.review_stars_x).round().value_counts()
distribution6 = (averuser_badbusiness.review_stars_x).round().value_counts()
distribution7 = (baduser_goodbusiness.review_stars_x).round().value_counts()
distribution8 = (baduser_averbusiness.review_stars_x).round().value_counts()
distribution9 = (baduser_badbusiness.review_stars_x).round().value_counts()

goodgood = generation(distribution1)
avergood = generation(distribution4)
toughgood = generation(distribution7)
goodbad = generation(distribution3)
averbad = generation(distribution6)
toughbad = generation(distribution9)
goodaver = generation(distribution2)
averaver = generation(distribution5)
toughaver = generation(distribution8)

goodrestaur_matrix = assign(goodgood,avergood,toughgood)
badrestaur_matrix = assign(goodbad,averbad,toughbad)
averrestaur_matrix = assign(goodaver,averaver,toughaver)
#goodrestaur_matrix = assign(goodgood,toughgood)
#badrestaur_matrix = assign(goodbad,toughbad)
#averrestaur_matrix = assign(goodaver,toughaver)
TestGraph(goodrestaur_matrix, averrestaur_matrix, badrestaur_matrix)

# for a special user
TURBO_DF = pd.read_csv(turbo_csv_filename)
TURBO_DF.head()
active_user = TURBO_DF['user_id'].value_counts()
# pick up a many review user
special_user = TURBO_DF[TURBO_DF['user_id'] == '0bNXP9quoJEgyVZu9ipGgQ']
special_goodbusiness = pd.merge(special_user, good_business, on = ['business_id', 'user_id'], how = 'inner')
special_averbusiness = pd.merge(special_user, average_business, on = ['business_id', 'user_id'], how = 'inner')
special_badbusiness = pd.merge(special_user, bad_business, on = ['business_id', 'user_id'], how = 'inner')
frames = [special_goodbusiness, special_averbusiness, special_badbusiness]
together = pd.concat(frames)
a = together['review_stars_x'].value_counts()
#print(a)
#print(generation(a))
