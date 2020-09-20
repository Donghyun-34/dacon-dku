import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('.\\data\\train.csv', index_col=0)
test = pd.read_csv('.\\data\\test.csv', index_col=0)
sample_submission = pd.read_csv('.\\data\\sample_submission.csv', index_col=0)

train_x = train.drop(columns='class', axis=1)
train_y = train['class']
test_x = test

forest = RandomForestClassifier(max_depth=1, random_state=0)
forest.fit(train_x, train_y)

y_pred = np.argmax(forest.predict_proba(test_x), axis=1)

submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission_1.csv', index=True)

print(forest.feature_importances_)