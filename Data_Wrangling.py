# Canonical Imports (Copy and paste these lines in your IPython notebooks for convenience)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
print(pd.__version__)
from IPython.core.display import HTML
print(os.environ['HOME'])

MY_YELP_JSON_RAW_DATA_DIR = '/Users/shuangwu/Desktop/yelp_dataset_challenge_academic_dataset'
MY_YELP_JSON_CLEAN_DATA_DIR = '/Users/shuangwu/Desktop/yelp_dataset_challenge_academic_dataset/Cleaned_Data_Directory'
MY_YELP_CSV_DATA_DIR = MY_YELP_JSON_CLEAN_DATA_DIR
MY_YELP_OUTPUT_DIR = MY_YELP_JSON_CLEAN_DATA_DIR

!mkdir -p $MY_YELP_JSON_CLEAN_DATA_DIR
!mkdir -p $MY_YELP_CSV_DATA_DIR
!echo MY_YELP_JSON_RAW_DATA_DIR : $MY_YELP_JSON_RAW_DATA_DIR
#!ls -sh $MY_YELP_JSON_RAW_DATA_DIR
#!echo
!echo MY_YELP_JSON_CLEAN_DATA_DIR : $MY_YELP_JSON_CLEAN_DATA_DIR
#!ls -sh $MY_YELP_JSON_CLEAN_DATA_DIR
#!echo
!echo MY_YELP_CSV_DATA_DIR : $MY_YELP_CSV_DATA_DIR
#!ls -sh $MY_YELP_CSV_DATA_DIR