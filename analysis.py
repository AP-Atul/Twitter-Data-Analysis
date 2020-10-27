import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from classifier.naive_bayes import NaiveBayes
from classifier.util import *

train = pd.read_csv("./dataset/train.csv")
test = pd.read_csv("./dataset/test.csv")
submission = pd.read_csv("./dataset/submission.csv")
train = train.drop(["id"], axis=1)
test = test.drop(["id"], axis=1)
submission = submission.drop(["id"], axis=1)

vectorizer = CountVectorizer(max_features=10, stop_words="english", max_df=0.7)

X_train = vectorizer.fit_transform(train["tweet"])
y_train = train["label"]
X_test = vectorizer.fit_transform(test["tweet"])

print(vectorizer.vocabulary_)

nb = NaiveBayes()
nb.fit(X_train.toarray(), y_train)
y_predictions = nb.predict(X_test.toarray())
print(y_predictions)

print(accuracy_score(submission["label"], y_predictions))
print(mean_squared_error(submission["label"], y_predictions))
