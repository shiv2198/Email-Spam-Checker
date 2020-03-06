import pandas as pd
import numpy as np
import os
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

def readFiles(path):
    for root, dirname, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            
            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1') 
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            
            f.close()
            message = '\n'.join(lines)
            yield path, message
                    

def dataFrameFromDirectory(path , classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message , 'class': classification})
        index.append(filename)
        
    return  pd.DataFrame(rows , index = index) 
    
dataset = pd.DataFrame({'message' : [] , 'class' : []})

dataset = dataset.append(dataFrameFromDirectory('D:/SHIVANSH/Machine Learning/Machine_Learning_AZ/email spam cheaker/emails/spam', 'spam'))
dataset = dataset.append(dataFrameFromDirectory('D:/SHIVANSH/Machine Learning/Machine_Learning_AZ/email spam cheaker/emails/ham', 'ham'))


#Bag of words model 
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(dataset['message'].values).toarray()
y = (dataset['class'].values)


#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Naive bayse Classification of spams
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#performance
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])
precision = cm[1][1]/(cm[0][1]+cm[1][1])
recall = cm[1][1]/(cm[1][1]+cm[1][0])
F1_score = 2*precision*recall/(precision+recall)









    