import sys
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path


def load_data(messages_filepath, categories_filepath):
    """
    Load the data and concatenate it
    Args:
        message_filepath: the file path of the messages.csv
        categories_filepath: the file path of the categories.csv
    Returns:
        concatenated df
    """
    # load messages dataset
    messages_filepath = Path(messages_filepath)
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories_filepath = Path(categories_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    messages.set_index("id", inplace=True)
    categories.set_index("id", inplace=True)
    df = messages.join(categories)

    return df


def clean_data(df):
    """
    The concatenated messages and categories data is cleaned.
    Categories are splitted and duplicates are deleted.
    Args:
        concatenated df of messages and categories
    Returns:
        concatenated df of messages with splitted categories, without doubles
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)

    # rename the columns of `categories`
    categories.columns = [
        categories.iloc[0][col].split("-")[0]
        for col in range(categories.shape[1])  # noqa
    ]

    # refactor the data to 1/0
    categories = categories.apply(lambda x: x.str.split("-").str[1])

    # Change to ints and set to binary
    categories = categories.astype(int)
    categories.replace(2, 1, inplace=True)

    # drop the categories column and join the cleaned data
    df.drop(columns="categories", inplace=True)
    df = df.join(categories)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the dataframe as a pickle
    Args:
        df: the DataFrame to be saved
        database_filename: The name as how to call the DataFrame
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("DisasterResponse", engine, index=False, if_exists="replace")

    return None


def main():
    """
    Run the main file
    Args are inputted by terminal
    """
    udacity = False
    if not udacity:
        messages_filepath, categories_filepath, database_filepath = (
            "data/disaster_messages.csv",
            "data/disaster_categories.csv",
            "data/DisasterResponse.db",
        )
    else:
        if len(sys.argv) == 4:

            messages_filepath, categories_filepath, database_filepath = sys.argv[
                1:
            ]  # noqa

    if messages_filepath:
        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
