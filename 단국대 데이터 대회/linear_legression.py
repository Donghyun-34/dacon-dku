import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('.\\data\\train.csv', index_col=0)
test = pd.read_csv('.\\data\\test.csv', index_col=0)
sample_submission = pd.read_csv('.\\data\\sample_submission.csv', index_col=0)


skf = StratifiedKFold(n_splits=4, shuffle=False, random_state=None)

train_x = train.drop(columns='class', axis=1)
train_y = train['class']
test_x = test


model = LogisticRegression()
model.fit(train_x, train_y)
p = model.predict(test_x)
int_p = p.astype(np.int64)
print(len(int_p))

submission = pd.DataFrame(int_p, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission_linear_version.csv', index=True)