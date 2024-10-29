Title of Project
Financial News Classification Using Random Forests

Objective
The goal of this project is to classify financial market news articles to determine sentiment or relevance, providing insights that can support financial decision-making or sentiment analysis.

Data Source
The dataset is available from the YBI Foundation's GitHub repository. It consists of financial news articles with labeled sentiments.

Import Library
python
Copy code
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
Import Data
python
Copy code
url = 'https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Financial%20Market%20News.csv'
df = pd.read_csv(url, encoding="ISO-8859-1")
Describe Data
Inspect the dataset to understand its structure and content:

python
Copy code
df.head()      # View the first few rows
df.info()      # Overview of columns and data types
df.shape       # Check dimensions of the dataset
df.columns     # List column names
Data Visualization
Basic visualizations (optional, requires additional libraries like matplotlib or seaborn) could include:

Distribution of target labels (e.g., Label column)
Word cloud for high-frequency terms
Bar plots for average word count per article
Example:

python
Copy code
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Label', data=df)
plt.title('Distribution of Labels')
plt.show()
Data Preprocessing
Concatenate text columns (from column 2 to 26) into a single string for each article:

python
Copy code
news = [' '.join(str(x) for x in df.iloc[i, 2:27]) for i in range(len(df))]
Define Target Variable (y) and Feature Variables (X)
Define y as the Label column and X as the processed news articles.

python
Copy code
X = news
y = df['Label']
Train Test Split
Split the data into training and testing sets:

python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2529)
Modeling
Vectorize the text data and train a Random Forest model:

python
Copy code
cv = CountVectorizer(lowercase=True, ngram_range=(1,1))
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train_cv, y_train)
Model Evaluation
Evaluate the model’s performance on the test data:

python
Copy code
y_pred = rf.predict(X_test_cv)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
Prediction
Use the model to make predictions on new data:

python
Copy code
sample_news = ["Your sample news article text here"]
sample_news_cv = cv.transform(sample_news)
prediction = rf.predict(sample_news_cv)
print("Prediction:", prediction)
Explanation
This project demonstrates a machine learning workflow to classify financial news articles. By preprocessing text, converting it into numerical features, and training a Random Forest model, we can predict the sentiment or relevance of unseen news data, which may aid financial analysis. Further improvements could include hyperparameter tuning, trying different vectorization techniques, or using other classification algorithms for better accuracy.
