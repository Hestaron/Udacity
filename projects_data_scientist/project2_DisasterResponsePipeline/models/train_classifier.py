# import sys
import pickle
import nltk
from pathlib import Path

nltk.download(
    ["punkt", "wordnet", "averaged_perceptron_tagger", "tagsets", "stopwords"]
)

import re
import numpy as np
import pandas as pd
import sys

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """Load the data from the SQL lite database"""
    engine = create_engine(f"sqlite:///{database_filepath}")
    sql_query = "SELECT * FROM DisasterResponse"
    #     try:
    df = pd.read_sql(sql_query, engine)
    #     except Exception as e:
    #         sql_query = "Select * FROM sqlite_master WHERE type='table'"
    #         print(pd.read_sql(sql_query, engine))
    X = df.message
    category_names = [
        "related",
        "request",
        "offer",
        "aid_related",
        "medical_help",
        "medical_products",
        "search_and_rescue",
        "security",
        "military",
        "child_alone",
        "water",
        "food",
        "shelter",
        "clothing",
        "money",
        "missing_people",
        "refugees",
        "death",
        "other_aid",
        "infrastructure_related",
        "transport",
        "buildings",
        "electricity",
        "tools",
        "hospitals",
        "shops",
        "aid_centers",
        "other_infrastructure",
        "weather_related",
        "floods",
        "storm",
        "fire",
        "earthquake",
        "cold",
        "other_weather",
        "direct_report",
    ]
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
    url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    text = re.sub(url_regex, "urlplaceholder", text)

    # tokenize
    text = re.sub(r"[^A-Za-z]", " ", text.lower())
    tokens = text.split(" ")

    # nltk stopwords + urlplaceholder
    tokens = [
        word
        for word in tokens
        if word not in stopwords.words("english") + ["urlplaceholder"]
    ]

    tokens_tagged = nltk.pos_tag(tokens)
    words = [
        word
        for word, tag in tokens_tagged
        if tag
        in [
            "JJ",
            "JJR",
            "JJS",  # Adjectives
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",  # Verbs
            "NN",
            "NNP",
            "NNPS",
            "NNS",  # Nouns
            "RB",
            "RBR",
            "RBS",  # Adverbs
        ]
    ]

    clean_tokens = [
        WordNetLemmatizer().lemmatize(w, pos="v") for w in words
    ]  # v for verbs
    return clean_tokens


def build_model():
    """
    Creating the pipeeline with countvectorization,
    TF-IDF transformation and randomforestclassification with a multioutputclassifier
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    parameters = {
        #         "clf__estimator__n_estimators": [25, 50, 100],
        "clf__estimator__min_samples_split": [2, 3],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
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

    Y_pred = model.predict(X_test)

    report = pd.DataFrame(
        data=[], index=["precision", "recall", "f1"], columns=category_names
    )

    for col in category_names:
        test_col = Y_test[col].reset_index(drop=True).rename("test")
        pred_col = pd.DataFrame(Y_pred, columns=Y_test.columns)[col].rename("predict")
        df_scores = pd.DataFrame([pd.to_numeric(test_col), pd.to_numeric(pred_col)]).T
        tp = np.where((df_scores.test == 1) & (df_scores.predict == 1), 1, 0).sum()
        fp = np.where((df_scores.test == 0) & (df_scores.predict == 1), 1, 0).sum()
        fn = np.where((df_scores.test == 1) & (df_scores.predict == 0), 1, 0).sum()

        if tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

        report.loc["precision"][col] = precision
        report.loc["recall"][col] = recall
        report.loc["f1"][col] = f1
    print(report)
    save_model(report, "models/metrics.pkl")
    return report


def save_model(model, model_filepath):
    """ "Save the model as a pickle in model_filepath"""
    with open(Path(model_filepath), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    udacity = True
    if udacity:
        if len(sys.argv) == 3:
            database_filepath, model_filepath = sys.argv[1:]
    else:
        database_filepath, model_filepath = (
            "data/DisasterResponse.db",
            "models/classifier.pkl",
        )
    print("Loading data...\n    DATABASE: {}".format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print("Building model...")
    model = build_model()

    print("Training model...")
    model.fit(X_train, Y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, Y_test, category_names)

    print("Saving model...\n    MODEL: {}".format(model_filepath))
    save_model(model, model_filepath)

    print("Trained model saved!")


if __name__ == "__main__":
    main()
