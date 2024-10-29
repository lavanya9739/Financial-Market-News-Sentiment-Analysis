# Financial-Market-News-Sentiment-Analysis
📈

**Table of Contents**
- Project Overview
- Dataset
- Installation
- Usage
- Project Structure
- Code Walkthrough
- Results
- Future Work
- License
  
**Project Overview**
- This project uses Natural Language Processing (NLP) techniques to classify financial market news articles into categories. By applying a Random Forest classifier, the model identifies sentiments or relevance within the news data, potentially useful for financial sentiment analysis.

**Dataset**
- The dataset is publicly available on YBI Foundation's GitHub repository and includes various financial news articles labeled for classification.

**File:**
- Financial Market News.csv
- Encoding: ISO-8859-1
  
**Installation**
- To run this project, clone the repository and install the required libraries:
- cd financial-news-classification
- pip install -r requirements.txt

**Add the following to a requirements.txt file:**
- txt
- numpy
- pandas
- scikit-learn
  
**Usage**
- Run the classification script in a Jupyter notebook or Python environment:
- Open the Jupyter notebook or script file.
- Execute each code block to load data, preprocess it, train the model, and evaluate the results.

**Project Structure**
financial-news-classification/
- ├── Financial_Market_News_Classification.ipynb   # Main notebook with code
- ├── README.md                                    # Project documentation
- └── requirements.txt                             # Python dependencies
  
**Code Walkthrough**

**1. Data Loading**
- The dataset is read into a pandas DataFrame with special encoding for compatibility.
  
**2. Data Preprocessing**
- Text Aggregation: The news columns are concatenated row-wise to form a single text string per article.
- Vectorization: CountVectorizer is applied to create a document-term matrix, representing text in numeric form.
  
**3. Feature Extraction**
- CountVectorizer is configured with:
- lowercase=True: Converts all text to lowercase for uniformity.
- ngram_range=(1,1): Unigrams only (single words as features).
  
**4. Model Training**
- Data is split into training (70%) and testing (30%) sets.
- RandomForestClassifier with 200 estimators is used to classify the data.
  
**5. Model Evaluation**
- Results are evaluated with a confusion matrix, classification report, and accuracy score.
  
**Results**
- The classification performance is summarized by:
- Confusion Matrix: Displays the true vs. predicted labels.
- Classification Report: Provides precision, recall, and F1-score for each label.
- Accuracy Score: The overall model accuracy.
  
**Future Work**
- Hyperparameter Tuning: Adjust parameters for improved accuracy.
- Model Comparison: Experiment with alternative classifiers, like Support Vector Machines or Gradient Boosting.
  
**License**
- This project is licensed under the MIT License. See the LICENSE file for details.
