import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'tagsets'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier

# load data from database
def load_data():
    """Load the data from the SQL lite database"""
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    sql_query = "SELECT * FROM DisasterTweets"
    df = pd.read_sql(sql_query, engine)
    X = df.message
    category_names = ['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']
    y = df[category_names]
    return X, y, category_names

def tokenize(text):
    """
    Tokenization of the text. Includes:
    -url replacing
    -tokenization
    -all lower text
    -removing stopwords
    -selecting adjectives, verbs, nouns and adverbs
    -Lemmatization
    
    Args:
    text: str text to be tokenized
    
    Returns:
    list of cleaned tokens
    """ 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)

    # tokenize
    text = re.sub(r"[^A-Za-z]", " ", text.lower())
    tokens = text.split(" ")
    
    # nltk stopwords + urlplaceholder
    stopwords_new = stopwords.words('english')
    tokens = [word for word in tokens if word not in stopwords.words('english')+ ['urlplaceholder']]
    
    tokens_tagged = nltk.pos_tag(tokens)
    words = [word for word, tag in tokens_tagged if tag in ["JJ", "JJR", "JJS", # Adjectives
                                                            "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", # Verbs
                                                            "NN", "NNP", "NNPS", "NNS", #  Nouns
                                                            "RB", "RBR", "RBS", # Adverbs
                                                            ]]
    
    clean_tokens = [WordNetLemmatizer().lemmatize(w, pos="v") for w in words] # v for verbs
    return clean_tokens

def build_model():
    """Creating the pipeeline with countvectorization, TF-IDF transformation and randomforestclassification with a multioutputclassifier"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline

def show_results(y_test, y_pred):
    """
    Create a function that shows the precision, recall and f1. 
    SKlearn package is not used, because it was not working properly.
    Displays the measures in an DataFrame.
    
    Args:
    y_test: pd.DataFrame of the ground truth
    y_pred: numpy list of the predicted outcomes
    
    Returns:
    pd.DataFrame containing the precision, recall and f1 score
    
    """
    report = pd.DataFrame(data=[], index=['precision','recall','f1'], columns=y_test.columns)    

    for col in y_test.columns:
        test_col = y_test[col].reset_index(drop=True).rename("test")
        pred_col = pd.DataFrame(y_pred, columns=y_test.columns)[col].rename("predict")
        df_scores = pd.DataFrame([pd.to_numeric(test_col), pd.to_numeric(pred_col)]).T
        tp = np.where((df_scores.test==1) & (df_scores.predict==1), 1, 0).sum()
        fp = np.where((df_scores.test==0) & (df_scores.predict==1), 1, 0).sum()
        fn = np.where((df_scores.test==1) & (df_scores.predict==0), 1, 0).sum()
        
        if tp==0:
            precision=0
            recall=0
            f1=0
        else:
            precision = tp / ( tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision*recall) / (precision + recall)

        report.loc['precision'][col] = precision
        report.loc['recall'][col] = recall
        report.loc['f1'][col] = f1
    display(report)
    return report