import re,sklearn,string
import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier as XGBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from nltk.stem import PorterStemmer
from string import punctuation

ps = PorterStemmer()

def readAbbrFile(abbrFile,abbr_dict):
    f = open( abbrFile )
    lines = f.readlines()
    f.close()
    for i in lines:
        tmp = i.split( '|' )
        abbr_dict[tmp[0]] = tmp[1]
    return abbr_dict


def convertCamelCase(word):
    return re.sub("([a-z])([A-Z])","\g<1> \g<2>",word)


def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def removeHash(s):
    return re.sub(r'#([^\s]+)', r'\1', s)


def cleanhtml(raw_html):
    cleanr = re.compile('<a>|</a>|<e>|</e>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def isNan(num):
    return num != num


def replaceAbbr(s,abbr_dict):
    for word in s:
        if word.lower() in abbr_dict.keys():
            s = [abbr_dict[word.lower()] if word.lower() in abbr_dict.keys() else word for word in s]
    return s


def pre_process(l,stopwords,abbr_dict ):
    l = cleanhtml ( l )
    l = replaceAbbr ( l,abbr_dict )
    l = removeHash ( l )
    l = re.sub ( '((www\.[\s]+)|(https?://[^\s]+))', '', l )
    l = re.sub ( r'\\[xa-z0-9.*]+', '', l )
    l = convertCamelCase ( l )
    l = replaceTwoOrMore ( l )

    #removing white spaces
    l = re.sub ( r'^RT[\s]+', '', l, flags=re.MULTILINE )  # removes RT

    # Removing usernames
    l = re.sub ( '@[^\s]+', '', l )

    # Removing words that start with a number or a special character
    l = re.sub ( r"^[^a-zA-Z]+", ' ', l )

    l = re.sub ( '[\s]+', ' ', l )

    # Removing words that end with digits
    l = re.sub ( r'\d+', '', l )

    # Replace the hex code "\xe2\x80\x99" with single quote
    l = re.sub ( r'\\xe2\\x80\\x99', "'", l )

    # Removing punctuation
    exclude=set(string.punctuation)
    l = ''.join ( ch for ch in l if ch not in exclude )

    # Remove trailing spaces and full stops
    l = l.strip ( ' .' )

    # Convert everything to lower characters
    l = l.lower ()
    
    tokens = re.split(' ',l)
    
    temp = []
    temp_tweet = ""
    for t in tokens:
        if(t!='' and t.lower() not in stopwords):
            ps.stem(t)
            temp.append(t)
            temp_tweet=temp_tweet+t+" "
    return temp_tweet
    
def train_preprocess(tweets,Classes,stopwords,abbr_dict ):
    mylist= []
    classes = []
    for l,m in zip(tweets,Classes):
        if(isNan(l)):
            continue
        if(type(m)!=int or not(m==0 or  m==1 or m==-1) ):
            continue
        temp_tweet = pre_process(l,stopwords,abbr_dict )
        classes.append(m)
        mylist.append(temp_tweet) 
    left = np.array(mylist)
    right = np.array(classes)
    return left,right


def test_preprocess(test_data,stopwords,abbr_dict ):
    test = []
    for t in test_data:
        t = t[0]
        temp_tweet = pre_process(t,stopwords,abbr_dict )
        test.append(temp_tweet)
    test = np.array(test)
    return test


def classify():
    linearSVM= LinearSVC( random_state=666, class_weight="balanced", max_iter=5000,  C=2.0,tol=0.001, dual=True )
    linearSVM_SVC= SVC( C=1, kernel="rbf", tol=1, random_state=0,gamma=1 )
    logistic = LogisticRegression( fit_intercept=True,class_weight="balanced", n_jobs=-1, C=1.0,
                                   max_iter=200 )
    rand_forest = RandomForestClassifier( n_estimators=403, random_state=666, max_depth=73, n_jobs=-1 )
    bc=BaggingClassifier( base_estimator=logistic, n_estimators=403, n_jobs=-1, random_state=666,
                                                max_features=410)

    ensemble_voting=VotingClassifier([("logistic",logistic),("rand_forest",rand_forest),("sgdc",SGDClassifier())],weights=[1,1,2])
    boost = AdaBoostClassifier(base_estimator=logistic)
    xgboost= XGBoostClassifier( n_estimators=103, seed=666, max_depth=4, objective="multi:softmax" )
    return ensemble_voting


def write_result(filename,predicted):
    q=0
    file = open(filename, "w")
    for a in predicted:
        q=q+1
        file.write(str(q)+";;"+str(a)+"\n")
    file.close()
