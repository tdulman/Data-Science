#!/usr/bin/env python
# coding: utf-8

# In[137]:


# import libraries
import nltk
import re
import sklearn
nltk.download('punkt')
nltk.download('wordnet')
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
import pickle
import sys
sys.path.insert(0,'../data')
sys.path.insert(0,'../models')
sys.path.insert(0,'../')
import os
ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
print(ROOT_DIR)
import sys
# print('The nltk version is {}.'.format(nltk.__version__))
# print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[ ]:





# In[138]:


# load data from database
def load_data():

    engine = create_engine("sqlite:///DisasterResponse.db".format(ROOT_DIR))


    df = pd.read_sql("SELECT * FROM {}".format('Disaster_Table'), engine)
    
    X = df['message']
    Y = df
    Y = Y.drop(columns = ['id','genre','categories','message','original'], axis=1)
    Y= Y.astype(int)
    return X, Y
import os
old_pwd = os.getcwd()
os.chdir("/home/workspace/data")
X,Y = load_data()
os.chdir(old_pwd)

print(Y)


# In[139]:


# Tokenize text in a df series object
def tokenize(df_series):
    """Tokenizes a df text series
    Args:
        df text series object
    Returns:
        clean token list
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = []
    for tok in df_series:
        clean = tokenizer.tokenize(tok)
        tokens.append(clean)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(str(tok)).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# In[140]:


def build_model(classifier):
    pipeline = Pipeline(
        [('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(classifier))]
        )

    return pipeline
# print(build_model(AdaBoostClassifier))
built_mdl = build_model(RandomForestClassifier())
built_mdl


# In[141]:


def train_model(X, Y, model):

    X_train, X_test, y_train, y_test = train_test_split(X, Y)
#     print(y_train.values)
    model_fit = model.fit(X_train, y_train)
    y_pred = model_fit.predict(X_test)

    return X_train, X_test, y_train, y_test, y_pred, model_fit

X_train, X_test, y_train, y_test, y_pred, model_fit = train_model(X,Y,built_mdl)


# In[142]:


def get_results(y_test, y_pred):

    results = pd.DataFrame(columns=['category', 'precision', 'recall', 'f_score'])
    count = 0
    for category in y_test.columns:
        precision, recall, f_score, support = score(y_test[category], y_pred[:,count], average='weighted')
        results.at[count+1, 'category'] =category
        results.at[count+1, 'precision'] = precision
        results.at[count+1, 'recall'] = recall
        results.at[count+1, 'f_score'] = f_score
        count += 1
    avg_precision = results['precision'].mean()
    print('Average precision:', avg_precision)
    print('Average recall:', results['recall'].mean())
    print('Average f_score:', results['f_score'].mean())
    return results
pred_results = get_results(y_test, y_pred)
pred_results


# In[143]:


def grid_search(model, X_train, y_train):

    param = {
            'clf__estimator__n_estimators': [100, 200],
            'vect__max_df': (0.5, 0.75, 1.0)
        }

    cv = GridSearchCV(model, param_grid=param, verbose = 2, n_jobs = -1)
    cv.fit(X_train.values, y_train.values)

    print("\nBest Parameters_rf:", cv.best_params_)
    print("Best cross-validation_rf score: {:.2f}".format(cv.best_score_))
    print("Best cross-validation score_rf: {}".format(cv.cv_results_))

    return cv

point_cv = grid_search(built_mdl,X_train,y_train)


# In[144]:


def save_cv(cv_name, cv):

    with open('{}.pickle'.format(cv_name), 'wb') as f:
        pickle.dump(cv, f)
cv_name = 'classifier'
save_cv(cv_name, point_cv)
    


# In[149]:


def save_clf_results(results_name, results):
 
    with open('{}.pickle'.format(results_name), 'wb') as f:
        pickle.dump(results, f)
results_name = 'ML_Results'
save_clf_results(results_name, pred_results)


# In[150]:


def save_model(model_name, model):

    with open('{}.pickle'.format(model_name), 'wb') as f:
        pickle.dump(model, f)
model_name = 'ML_Model'
save_model(model_name,model_fit )


# In[ ]:




