import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
)  # noqa
from sklearn.preprocessing import OneHotEncoder


def fill_median(data, cols, fresh):
    """
    Fill the medians of missing columns and create an extra column with
    boolean values. True if the value was missing in the original.
    Args:
        data: DataFrame, where the columns are in
        cols: str or list, the columns where the median needs to be filled
        fresh: Boolean, if you want new pickles to be created for the medians.
    Returns:
        DataFrame, with no missing values in the passed cols and for every
        cols an extra column.
    """
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        data[f"{col}_missing"] = (data[col] > 0) is False

        # Save for interface
        if fresh:
            median = data[col].median()
            save_pickle(median, f"data/median_{col}.pkl")
        else:
            median = load_pickle(f"data/median_{col}.pkl")

        data[col].fillna(median, inplace=True)
    return data


def save_pickle(var, file):
    """Save a 'var' with name 'file'."""
    with open(file, "wb") as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None


def load_pickle(file):
    """Load a pickle with 'file' as Path"""
    with open(file, "rb") as pkl_file:
        pkl = pickle.load(pkl_file)
    return pkl


def oh_encoding(
    data,
    bin_cols=["income_missing", "age_missing"],
    cat_cols=["gender", "offer_type"],  # noqa
):
    """
    One-Hot Encoder for the binary and categorical columns
    Args:
        data: DataFrame
        bin_cols: list, column names where numbers needs to be encoded
        cat_cols: list, column names where categorical need to be encoded
    Return:
        DataFrame
    """
    data[bin_cols] = data[bin_cols].astype(int)

    encoder = OneHotEncoder()
    encoder.fit(data[cat_cols])

    cols = [
        feature.replace("x0", "gender").replace("x1", "offer")
        for feature in encoder.get_feature_names()
        # Change to get_feature_names_out() in newest scikit-learn
    ]
    df = pd.DataFrame(
        encoder.fit_transform(data[cat_cols]).toarray(),
        columns=cols,
        index=data.index,  # noqa
    )
    # df.set_index(data.index)
    data.drop(columns=["gender", "offer_type"], inplace=True)
    data = data.join(df)
    return data


def print_scores(y_test, y_pred):
    """
    Prints metrics
    Args:
        y_test: Series, the real target values
        y_pred: Series, the predicted target values
    Returns:
        Accuracy, Recall, Precision and F1-score"""
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("The scikit-learn RandomForests model metrics:")
    print(f"Accuracy: {acc}")
    print(f"Recall: {rec}")
    print(f"Precision: {precision}")
    print(f"f1-score: {f1}")
    return acc, rec, precision, f1


def predict_offer(
    data,
):
    """
    Predicts the whether the data suggests that somebody will accept the offer.
    """
    try:
        classifier = load_pickle("model/rf_tuned.pkl")
    except KeyError:
        classifier = load_pickle("model/rf.pkl")
    except BaseException:
        raise (
            "Make sure the model is created through using 'create_model.py' and if you are in the right directory."
        )

    pred = classifier.predict(data)
    return int(pred)


def create_show_data_buttons(data):
    """
    Creates buttons to show concise information about the offer data
    or about the person
    """
    if st.button("Show data offer"):
        st.dataframe(
            data[
                [
                    "reward",
                    "difficulty",
                    "duration",
                    # "offer_type",
                    "email",
                    "mobile",
                    "social",
                    "web",
                ]
            ]
        )
    if st.button("Show data person"):
        st.dataframe(data[["age", "gender", "income"]])
    return None


def create_predict_button(data):
    """
    Creates the button to predict if the parameters suggest
    that the offer will be accepted.
    """
    # The predict button
    if st.button("Predict if the offer will be completed"):
        output = predict_offer(data)
        if output == 0:
            prediction_message = (
                "The customer is estimated to not complete the offer."  # noqa
            )
        elif output == 1:
            prediction_message = (
                "The customer is estimated to complete the offer."  # noqa
            )
        else:
            raise BaseException("Invalid output predicted from the model.")

        st.success(prediction_message)
    return None
