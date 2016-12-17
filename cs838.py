import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import rpy2
import pylab
import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *
py.sign_in('swu5656', 'zxcvbn')
pd.options.display.mpl_style = 'default'
print pd.__version__

def importData():
    #CREATE DIRECTORY
    MY_YELP_DATA_DIR = '/Users/shuangwu/Desktop/test/'
    MY_YELP_CLEAN_DATA_DIR = '/Users/shuangwu/Desktop/test/Cleaned_Data_Directory'
    #READ DATA INTO PANDAS DATAFRAM
    final_csv_filename = os.path.join(MY_YELP_DATA_DIR, 'final.csv')
    final_DF = pd.read_csv(final_csv_filename)
    return final_DF

# GRAPH 1:
def graph1(final_DF):
    N_star_categories = 5
    colors = np.array(['#E50029', '#E94E04', '#EEC708', '#A5F30D', '#62F610']) # 1, 2, 3, 4, and 5 stars respectively
    stars_labels = np.array([x_stars+1 for x_stars in range(N_star_categories)])
    star_category_dist_fig = plt.figure(figsize=(12,8))
    bar_plot_indices = np.arange(N_star_categories)
    star_category_absolute_frequencies = final_DF.review_stars.value_counts(ascending=True)
    star_category_relative_frequencies = np.array(star_category_absolute_frequencies)/float(sum(star_category_absolute_frequencies))
    rects = plt.bar(bar_plot_indices, star_category_relative_frequencies, width=1, color=colors, alpha=.2)
    for (idx, rect) in enumerate(rects):
        plt.gca().text(rect.get_x()+rect.get_width()/2., 1.05*rect.get_height(), '%d'%int(star_category_absolute_frequencies[idx+1]), ha='center', va='bottom')
    plt.xticks(bar_plot_indices+0.5, stars_labels, rotation='horizontal')
    #x, labels, rotation='vertical'
    plt.xlabel('Star Category')
    plt.ylabel('Relative Frequency')
    plt.ylim([0,1])
    plt.title('Star Category Distribution for {0} Reviews'.format(len(final_DF)))
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('test.png')

#'''
#GRAPH 2:
def graph2(final_DF):
    company = 'auESFwWvW42h6alXgFxAXQ'
    #final_DF = final_DF.set_index(['business_id', 'date'])
    #gbt = final_DF.groupby(['business_id', 'date'])
    meanGbt = final_DF.groupby(['business_id', 'date']).mean()
    businessStarOverview= meanGbt.iloc[ 0, :]
    businessStarOverview= meanGbt
    plt.plot(np.array([x + 1 for x in range(businessStarOverview.count())]), businessStarOverview)
    plt.axis([0, businessStarOverview.count() +1 , 0, 10])

    #plt.xticks(businessStarOverview.index.levels[1], stars_labels, rotation='horizontal')
    #x, labels, rotation='vertical'
    plt.xlabel('Star Category')
    plt.ylabel('Relative Frequency')
    plt.ylim([0,10])
    plt.title('Star Category Distribution for {0} Latest Comments Days'.format(len(businessStarOverview)))
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('test2.png')
#'''

def main():
    final_DF = importData()
    graph1(final_DF)
    #graph2(final_DF)


if __name__ == "__main__":
        main()













