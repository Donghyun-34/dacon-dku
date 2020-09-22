import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

train = pd.read_csv('.\\data\\train.csv', index_col=0)
test = pd.read_csv('.\\data\\test.csv', index_col=0)
sample_submission = pd.read_csv('.\\data\\sample_submission.csv', index_col=0)

model = AdaBoostClassifier(n_estimators=1000, random_state=0)

train_x = train.drop(columns='class', axis=1)
train_y = train['class']
test_x = test
model.fit(train_x, train_y)
predict_data = model.predict(test_x)

submission = pd.DataFrame(data=predict_data, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission_1.csv', index=True)