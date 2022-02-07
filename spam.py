from numpy import reciprocal
import pandas as pd
import pickle

data = pd.read_csv("dataset_sms_spam_v1.csv", sep=",")

X = data.iloc[:,0]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_train,y_train)


pickle.dump(classifier, open('spam.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizr.pkl', 'wb'))
