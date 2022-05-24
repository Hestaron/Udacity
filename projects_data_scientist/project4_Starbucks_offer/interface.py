import streamlit as st
import pandas as pd
import numpy as np
from create_model import preprocessing
from functions import create_show_data_buttons, create_predict_button


def create_sidebar():
    """
    Create the sidebar parameter inputs.
    Some offers are pre-defined according to the data.
    They can also be selected, hence the big if/else construction.
    Return:
        The parameter values of the inputs.
    """

    st.sidebar.header("The parameters")
    time_offer_received = st.sidebar.slider(
        label="Time offer received", min_value=0, max_value=576, step=1
    )

    gender = st.sidebar.selectbox(
        "Gender", ("Male", "Female", "Other", "Unknown")
    )  # noqa
    # Transform gender
    gender_dict = {"Other": "O", "Male": "M", "Female": "F"}
    for key in gender_dict.keys():
        gender = gender.replace(key, gender_dict[key])

    age_missing = st.sidebar.checkbox(label="Age unknown")
    if age_missing:
        age = np.nan
    else:
        age = st.sidebar.slider(
            label="Age", min_value=18, max_value=99, value=25, step=1
        )  # noqa

    income_missing = st.sidebar.checkbox(label="Income unknown")
    if income_missing:
        income = np.nan
    else:
        income = st.sidebar.slider(
            label="Income ($)",
            min_value=0,
            max_value=120_000,
            value=80_000,
            step=1_000,  # noqa
        )

    days_member = st.sidebar.slider(
        label="Time a member (days)", max_value=1712
    )  # noqa

    offer = st.sidebar.selectbox(
        "Offer type",
        (
            "Custom",
            "Offer 0",
            "Offer 1",
            "Offer 2",
            "Offer 3",
            "Offer 4",
            "Offer 5",
            "Offer 6",
            "Offer 7",
            "Offer 8",
            "Offer 9",
        ),
    )
    if offer == "Custom":
        reward = st.sidebar.slider(
            label="Reward ($)", min_value=0, max_value=10, value=5, step=1
        )

        difficulty = st.sidebar.selectbox("Dollars to spent", (0, 5, 7, 10, 20))  # noqa

        duration = st.sidebar.selectbox(
            "Duration of offer (days)", (3, 4, 5, 7, 10)
        )  # noqa
        duration = duration * 24

        offer_type = st.sidebar.selectbox(
            "Type of Offer",
            ("informational", "discount", "buy one get one free"),  # noqa
        ).replace("buy one get one free", "bogo")

        email = st.sidebar.checkbox(label="Email offer", value=True)
        mobile = st.sidebar.checkbox(label="Mobile offer", value=True)
        social = st.sidebar.checkbox(label="Social Media offer", value=True)
        web = st.sidebar.checkbox("Web offer", value=True)
    elif offer == "Offer 0":
        reward = 10
        difficulty = 10
        duration = 7
        offer_type = "bogo"
        email = 1
        mobile = 1
        social = 1
        web = 0
    elif offer == "Offer 1":
        reward = 10
        difficulty = 10
        duration = 5
        offer_type = "bogo"
        email = 1
        mobile = 1
        social = 1
        web = 1
    elif offer == "Offer 2":
        reward = 0
        difficulty = 0
        duration = 4
        offer_type = "informational"
        email = 1
        mobile = 1
        social = 0
        web = 1
    elif offer == "Offer 3":
        reward = 5
        difficulty = 5
        duration = 7
        offer_type = "bogo"
        email = 1
        mobile = 1
        social = 0
        web = 1
    elif offer == "Offer 4":
        reward = 5
        difficulty = 20
        duration = 10
        offer_type = "discount"
        email = 1
        mobile = 0
        social = 0
        web = 1
    elif offer == "Offer 5":
        reward = 3
        difficulty = 7
        duration = 7
        offer_type = "discount"
        email = 1
        mobile = 1
        social = 1
        web = 1
    elif offer == "Offer 6":
        reward = 2
        difficulty = 10
        duration = 10
        offer_type = "discount"
        email = 1
        mobile = 1
        social = 1
        web = 1
    elif offer == "Offer 7":
        reward = 0
        difficulty = 0
        duration = 3
        offer_type = "informational"
        email = 0
        mobile = 1
        social = 1
        web = 1
    elif offer == "Offer 8":
        reward = 5
        difficulty = 5
        duration = 5
        offer_type = "bogo"
        email = 1
        mobile = 1
        social = 1
        web = 1
    elif offer == "Offer 9":
        reward = 2
        difficulty = 10
        duration = 7
        offer_type = "discount"
        email = 1
        mobile = 1
        social = 0
        web = 1
    data = pd.DataFrame(
        [
            [
                time_offer_received,
                gender,
                age,
                income,
                reward,
                difficulty,
                duration,
                offer_type,
                email,
                mobile,
                social,
                web,
                days_member,
            ]
        ],
        columns=[
            "time_offer_received",
            "gender",
            "age",
            "income",
            "reward",
            "difficulty",
            "duration",
            "offer_type",
            "email",
            "mobile",
            "social",
            "web",
            "days_member",
        ],
    )
    return data


def add_empty_cat_columns(data):
    """
    Some columns are missing which are necessary for the model to predict.
    They are added to the DataFrame here.
    Args:
        data: DataFrame
    Return:
        data: DataFrame, with the missing columns added with 0 as value
    """
    cat_cols = [
        "gender_M",
        "gender_F",
        "gender_O",
        "gender_Unknown",
        "offer_type_bogo",
        "offer_type_informational",
        "offer_type_discount",
    ]

    # add the categorical columns to the dataframe
    # such that predictions are possible
    cols_add = [cols for cols in cat_cols if cols not in data.columns]
    for c in cols_add:
        data[c] = 0

    return data


def main():
    """
    Create the interface as the main script.
    """

    st.title("Starbucks offer acceptation Prediction")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">
    Offer Acceptation Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create the sliders at the left side
    raw_data = create_sidebar()

    create_show_data_buttons(raw_data)

    # preprocess the data
    data = preprocessing(raw_data)

    # Add empty cols for the prediction
    data = add_empty_cat_columns(data)

    create_predict_button(data)

    return None


if __name__ == "__main__":
    main()
