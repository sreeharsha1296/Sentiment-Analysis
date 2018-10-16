import pandas,methods
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,precision_recall_fscore_support
from sklearn.model_selection import KFold
from methods import *

abbr_dict = {}
stopwords = {}

obama_training_data = pandas.read_excel('training-Obama-Romney-tweets.xlsx',sheet_name ="Obama",keep_default_na = False)
obama_test_data = pandas.read_excel('final-testData-no-label-Obama-Romney-tweets.xlsx',sheet_name ="Obama",keep_default_na = False,header=None,index_col=0)
obama_test_data = obama_test_data.as_matrix()

obama_train_tweets = obama_training_data['Anootated tweet'] 
obama_train_classes = obama_training_data['Class']

romney_training_data = pandas.read_excel('training-Obama-Romney-tweets.xlsx',sheet_name ="Romney",keep_default_na = False)
romney_test_data = pandas.read_excel('final-testData-no-label-Obama-Romney-tweets.xlsx',sheet_name ="Romney",keep_default_na = False,header=None,index_col=0)
romney_test_data = romney_test_data.as_matrix()

romney_train_tweets = romney_training_data['Anootated tweet'] 
romney_train_classes = romney_training_data['Class']

		
text_clf = Pipeline([('vect', CountVectorizer()),
			         ('tfidf', TfidfTransformer()),
			         ('clf',classify())])

with open("stop_words.txt", "r") as f:
    lines = f.read().splitlines()
for line in lines:
    stopwords[line] = 1

abbr_dict = readAbbrFile("abbr",abbr_dict)

left,right = train_preprocess(obama_train_tweets,obama_train_classes,stopwords,abbr_dict )
#Preprocessed data and their corresponding labels for Obama

obama_test = test_preprocess(obama_test_data,stopwords,abbr_dict )
text_clf = text_clf.fit(left,right)
predicted = text_clf.predict(obama_test)
write_result("Obama_Results.txt",predicted)

left,right = train_preprocess(romney_train_tweets,romney_train_classes,stopwords,abbr_dict )
#Preprocessed data and their corresponding labels for Romney

romney_test = test_preprocess(obama_test_data,stopwords,abbr_dict )
text_clf = text_clf.fit(left,right)
predicted = text_clf.predict(romney_test)
write_result("Romney_Results.txt",predicted)