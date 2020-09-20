from matplotlib import rcParams, pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from warnings import simplefilter

train = pd.read_csv('.\\data\\train.csv', index_col=0)
test = pd.read_csv('.\\data\\test.csv', index_col=0)
sample_submission = pd.read_csv('.\\data\\sample_submission.csv', index_col=0)

print(train.info())

train_x = train.drop(columns='class', axis=1)
train_y = train['class']
test_x = test