# Natural Language Processing

# Import library
import numpy as np
import pandas as pd
import matplotlib as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) 
# quoting = 3 ignore double quotes

 
# Cleaning the texts
# dataset['Review'][0] First review
import re
import nltk # Remove prepositions Doesn't help with 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i] ) # ^ Do not remove Replace with a space                    
    # Only keep the letters A-Z. Remove numbers periods and etc
    # Make everything lover case
    review = review.lower() # Lowercase
    review = review.split() # Split into elements in a list
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # Update list in forloop 
    # Faster to process as a set instead of a list
    review = ' '.join(review) # Join the elements in the list to a string
    corpus.append(review)

# Creating the Bag of words model

# Why do we need to 
# Need to predict whether review was positive or negative
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Train the model with machine learning
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Random Forest
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 500,
#                                    criterion = 'entropy',
#                                    random_state = 0)
#classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1_score = 2*precision*recall/(precision+recall)