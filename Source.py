import numpy as np


import pandas as pd

url = 'https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Financial%20Market%20News.csv'
df = pd.read_csv(url, encoding="ISO-8859-1")

df.head()


df.info()

df.shape

df.columns

' '.join(str(x) for x in df.iloc[1,2:27])

df.index

len(df.index)

news = []
for i in range(0,len(df.index)):
   news.append(' '.join(str(x) for x in df.iloc[i,2:27]))


type(news)

news[0]

X = news

type(X)

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(lowercase=True, ngram_range=(1,1))

X=cv.fit_transform(X)

X.shape

y = df['Label']

y.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2529)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
