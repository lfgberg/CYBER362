# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:10:08 2019

@author: jdk450
"""

import os
import pandas  as pd
from sklearn import model_selection, linear_model, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#Replace the above line with this one if you get an error
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import tree

# load the dataset
data = pd.read_csv('SMSSpamCollection.csv')

# create  dataframes using texts and labels
texts = data.iloc[:, 1]
labels = data.iloc[:, 0]

#take a look
texts
labels

#Use Tfidf Vectorizer as explained in Canvas
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# split the dataset into training and validation datasets 
X_train, X_test, y_train, y_test = train_test_split(X,labels, test_size=0.20,random_state=0)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

y_predict = decision_tree.predict(X_test)

#get confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_predict)

#show confusion matrix
print(cnf_matrix)

#Calculate performance measures:
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

#if you don't include pos_label='sham' you get this error:
#ValueError: pos_label=1 is not a valid label: array(['ham', 'spam'], dtype='<U4')
print("Precision:", metrics.precision_score(y_test, y_predict, pos_label='spam'))
print("Recall:",metrics.recall_score(y_test, y_predict, pos_label = 'spam'))
print("F1-score", metrics.f1_score(y_test, y_predict, pos_label='spam'))

#from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve ,recall_score,classification_report
print(metrics.classification_report(y_test, y_predict))

#create heatmap of confusion matrix
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# create heatmap
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


