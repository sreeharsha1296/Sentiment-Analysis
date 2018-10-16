# Twitter Sentiment Analysis

### Problem Statement
Given a collection of tweets, classify them into three classes namely: positive, negative, neutral. The tweets used here are pertaining to two US presidential candidates Barack Obama and Mitt Romney, the contestants of the 2012 US Presidential Election. By classifying the tweets into the mentioned classes we would be capable of predicting the opinion of the public and get a sense of the outcome of the election.

### Preprocessing
The following preprocessing steps have been performed on the data from twitter
1) Removal of Hyperlinks, Usernames
2) Split camel case words
3) Removal of annotations, hashtags and special characters
4) Stripping of white spaces
5) Removal of stopwords, digits
6) Tokenized and lemmatization of tweets

### Classification Methods
Various classification models have been tried and tested and the final model is an ensemble of Stochastic Gradient Descent Classifier, Random Forest and Logistic Regression

The following classification methods have been used to determine the best fit
1) Multinomial Naive Bayes Classifier
2) Support Vector Machines(SVM) (Linear and RBF Kernel)
3) Stochastic Gradient Descent(SGD)
4) Logistic Regression
5) Random Forest
6) Xgboost Classifier

###Evaluation
The evaluation of classifiers on training data has been done on the following metrics
1) Precision
2) Recall
3) F1-score

###Execution
Run the python file main.py to get test results of both Obama and Romney datasets. The results have been written to the two output files visible in the repository.