import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    """
    Tokenize the input text for the disaster response classification
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine("sqlite:///data/DisasterResponse.db")
# df = pd.read_sql_table("DisasterResponse", engine)
sql_query = "SELECT * FROM DisasterResponse"
df = pd.read_sql(sql_query, engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    """
    Load the index page and generate the data for the graphs below
    """

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    # create visuals
    # F1 visual
    metrics = pd.read_pickle("models/metrics.pkl")
    metrics = metrics.T.rename(
        columns={"f1": "F1-score", "precision": "Precision", "recall": "Recall"}  # noqa
    )

    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [
                Bar(
                    name="F1-score",
                    x=metrics["F1-score"],
                    y=metrics.index,
                    orientation="h",
                    # width=2,
                ),
                Bar(
                    name="Recall",
                    x=metrics["Recall"],
                    y=metrics.index,
                    orientation="h",
                    # width=2,
                ),
                Bar(
                    name="Precision",
                    x=metrics["Precision"],
                    y=metrics.index,
                    orientation="h",
                    # width=2,
                ),
            ],
            "layout": {
                "title": "F1-score per category",
                "xaxis": {"title": "F1-score"},
                "yaxis": {"title": "Categories"},
                "height": 1000,
                # "width": 0,
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    """
    Show the categorization of the tweet
    """
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
