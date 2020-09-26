from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from matplotlib import rcParams
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings

rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')
pd.set_option('max_columns', 100)
pd.set_option("display.precision", 4)
warnings.simplefilter('ignore')

train = pd.read_csv('.\\data\\train.csv', index_col=0)
test = pd.read_csv('.\\data\\test.csv', index_col=0)
sample_submission = pd.read_csv('.\\data\\sample_submission.csv', index_col=0)

traget_col = 'class'
seed = 42

train_x = train.drop(columns='class', axis=1)
train_y = train['class']
test_x = test


X_trn, X_val, y_trn, y_val = train_test_split(train_x, train_y, test_size=.2, random_state=seed)
print(X_trn.shape, X_val.shape, y_trn.shape, y_val.shape)

clf = LGBMClassifier(objective='multiclass',
                     n_estimators=1000,
                     num_leaves=64,
                     learning_rate=0.1,
                     min_child_samples=10,
                     subsample=.5,
                     subsample_freq=1,
                     colsample_bytree=.8,
                     random_state=seed,
                     n_jobs=-1)
clf.fit(X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        eval_metric='multiclass',
        early_stopping_rounds=10)
p_val = clf.predict(X_val)
p_tst = clf.predict(test)

print(f'{accuracy_score(y_val, p_val) * 100:.4f}%')