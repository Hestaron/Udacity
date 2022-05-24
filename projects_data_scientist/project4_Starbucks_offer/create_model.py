import pandas as pd
import numpy as np
from functions import fill_median, oh_encoding, print_scores, save_pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


def load_data():
    """
    Load the data and merge the data together.
    Perform some basic column selection and renaming.
    Returns:
        DataFrame
    """
    # Read in the json files
    portfolio_raw = pd.read_json(
        "data/portfolio.json", orient="records", lines=True
    )  # noqa
    profile_raw = pd.read_json(
        "data/profile.json", orient="records", lines=True
    )  # noqa
    transcript_raw = pd.read_json(
        "data/transcript.json", orient="records", lines=True
    )  # noqa

    # Portfolio
    portfolio = portfolio_raw.join(
        portfolio_raw.pop("channels").str.join("|").str.get_dummies()
    )

    # Split the json-like value column
    transcript = transcript_raw.join(transcript_raw.value.apply(pd.Series))

    # Merge the transcripts with the persons
    transcript_persons = transcript.merge(
        profile_raw, left_on="person", right_on="id", how="outer"
    )

    # Combine the offer_id and the 'offer id' columns
    transcript_persons["offer_id"] = transcript_persons["offer_id"].where(
        ~transcript_persons["offer_id"].isna(), transcript_persons["offer id"]
    )

    # Only select the relevant columns
    df = transcript_persons[
        [
            "event",
            "time",
            "offer_id",
            "amount",
            "gender",
            "age",
            "became_member_on",
            "income",
            "person",
        ]
    ]

    # merge the transcripts, persons and portfolio
    df_total = df.merge(
        portfolio, right_on="id", left_on="offer_id", how="outer"
    )[  # noqa
        [
            "event",
            "time",
            "amount",
            "gender",
            "age",
            "became_member_on",
            "income",
            "reward",
            "difficulty",
            "duration",
            "offer_type",
            "email",
            "mobile",
            "social",
            "web",
            "person",
            "id",
        ]
    ]

    # Rename id's for clarity
    df_total.rename(
        columns={"id": "transaction_id", "person": "person_id"}, inplace=True
    )  # noqa
    return df_total


def groupby_offer(data):
    """
    In order to analyse offers per row.
    The data is split in several DataFrames.
    Offer received, offer viewed and offer completed.
    Args:
        data: DataFrame
    Returns:
        DataFrame
    """
    # Duration in hours
    data.duration = data.duration * 24

    # Select the offers received, viewed and completed
    offer_received = data[data.event == "offer received"]
    offer_viewed = data[data.event == "offer viewed"][
        ["person_id", "transaction_id", "time"]
    ]
    offer_completed = data[data.event == "offer completed"][
        ["person_id", "transaction_id", "time"]
    ]

    # Add offers viewed to offers received
    df_offer = offer_received.merge(
        offer_viewed,
        left_on=["person_id", "transaction_id"],
        right_on=["person_id", "transaction_id"],
        how="outer",
    )
    df_offer = df_offer.rename(
        columns={"time_x": "time_offer_received", "time_y": "time_offer_viewed"}  # noqa
    )
    df_offer["time_offer_ended"] = (
        df_offer.time_offer_received + df_offer.duration
    )  # noqa

    # The offer is either seen within the period of receiving and ending
    # OR the offer is not seen at all (null value)
    df_offer = df_offer[
        (
            (df_offer.time_offer_received <= df_offer.time_offer_viewed)
            & (df_offer.time_offer_viewed <= df_offer.time_offer_ended)
        )
        | (df_offer.time_offer_viewed.isna())
    ]

    # Merging the offer_completed
    df_offer = df_offer.merge(
        offer_completed,
        left_on=["person_id", "transaction_id"],
        right_on=["person_id", "transaction_id"],
        how="outer",
    )
    df_offer.rename(columns={"time": "time_completed"}, inplace=True)
    data = df_offer[
        (
            (df_offer.time_offer_viewed <= df_offer.time_completed)
            & (df_offer.time_completed <= df_offer.time_offer_ended)
        )
        | (df_offer.time_completed.isna())
    ]
    return data


def preprocessing(data, fresh=False):
    """
    Preprocessing the data for modelling.
    Age outliers are corrected,
    Unnecessary columns are dropped,
    missing values in income and age are filled with medians,
    One hot encoding of categorical values takes place.

    Returns:
        DataFrame
    """
    # The age 118 is unlikely for so many people.
    # We assume that it is a default value of some sorts.
    data.age.where(data.age != 118, inplace=True)

    # Delete columns
    try:
        data = data.drop(
            columns=[
                "time_offer_viewed",
                "event",
                "amount",
                "person_id",
                "transaction_id",
            ]  # noqa
        )
    except KeyError:
        pass

    data = fill_median(data, ["income", "age"], fresh=fresh)

    # Check null values
    null_cols = data.columns[data.isna().sum() > 0]
    assert (
        data.isna().sum().sum() == 0
    ), f"There are null values in the data in the columns: {null_cols}"

    # One-Hot Encoding
    data = oh_encoding(data)
    return data


def get_param_grid():
    """
    Generate the parameter grid for hyper parameter tuning
    Returns:
        random_grid: DataFrame
    """
    # According to the different sources on the internet,
    # a RandomizedSearchCV would be the best approach.
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

    # Number of features to consider at every split
    max_features = ["auto", "sqrt"]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the param grid
    param_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }
    return param_grid


def create_model(X_train, y_train, tuned=False):
    """
    Create the classification model
    Args:
        X_train: DataFrame, the training data
        y_train: Series, the training data target
        tuned: Boolean, wether to tune the model or not
    Returns:
        machine learning model"""
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_train, y_train)

    if tuned:
        param_grid = get_param_grid()
        classifier = RandomizedSearchCV(
            estimator=classifier,
            param_distributions=param_grid,
            n_iter=100,
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )
        classifier.fit(X_train, y_train)
        save_pickle(classifier, "model/rf_tuned.pkl")
    else:
        save_pickle(classifier, "model/rf.pkl")

    return classifier


def main():
    """
    Train a model that predicts if a customer would complete an offer.
    """
    # Parameters
    test_size = 0.15
    seed = 42

    # Load the data
    data = load_data()

    # Transform the data such that offers are easier analysed
    data = groupby_offer(data)

    # Transform became_member_on to days member
    data.became_member_on = pd.to_datetime(
        data.became_member_on.astype(int), format="%Y%m%d"
    )
    data["days_member"] = (
        data.became_member_on.max() - data.became_member_on
    ).dt.days  # noqa
    data.pop("became_member_on")

    # Create target
    data["target"] = data["time_completed"] > 0
    data.drop(columns=["time_completed", "time_offer_ended"], inplace=True)

    float_cols = [
        "time_offer_received",
        "reward",
        "difficulty",
        "duration",
        "email",
        "mobile",
        "social",
        "web",
    ]
    data[float_cols] = data[float_cols].astype(int)

    # Fill gender
    data.gender.fillna("Unknown", inplace=True)

    # Categoricals
    cat_cols = ["gender", "offer_type"]
    data[cat_cols] = data[cat_cols].astype("category")

    # Train Test split
    X = data.drop(columns="target")
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    X_train = preprocessing(X_train, fresh=True)
    X_test = preprocessing(X_test)

    # Random Forest
    classifier = create_model(X_train, y_train, tuned=True)

    y_pred = classifier.predict(X_test)
    accuracy, recall, f1_score = print_scores(y_test, y_pred)

    return accuracy, recall, f1_score


if __name__ == "__main__":
    main()
