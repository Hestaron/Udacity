import os
import glob
import psycopg2
import pandas as pd
from sql_queries import *
import datetime as dt
from create_tables import main as create_tables


def process_song_file(cur, filepath):
    """
    Processes the song data and puts the data in the song and artist database.
    """
    # open song file
    df = pd.read_json(filepath, typ="series")

    # insert song record
    song_data = list(df[["song_id", "title", "artist_id", "year", "duration"]].values)
    cur.execute(song_table_insert, song_data)

    # insert artist record
    artist_data = list(
        df[
            [
                "artist_id",
                "artist_name",
                "artist_location",
                "artist_latitude",
                "artist_longitude",
            ]
        ].values
    )
    cur.execute(artist_table_insert, artist_data)


def process_log_file(cur, filepath):
    """
    Processes the log data.
    Only the data with property 'NextSong' is selected.
    Time in miliseconds is converted to the different time properties and put in the 'time' database.
    Then the user data is put in the database under 'users'.
    Read in data from songs and artists to create songplay.
    """
    # open log file
    df = pd.read_json(filepath, lines=True)

    # filter by NextSong action
    df = df.loc[df.page == "NextSong"]

    # convert timestamp column to datetime
    t = pd.to_datetime(df.ts * 1_000_000)

    # insert time data records
    time_data = [
        list(t.astype(str).str[:19]),
        list(t.dt.hour),
        list(t.dt.day),
        list(t.dt.week),
        list(t.dt.month),
        list(t.dt.year),
        list(t.dt.weekday),
    ]
    column_labels = ("timestamp", "hour", "day", "week", "month", "year", "weekday")
    time_df = pd.DataFrame.from_records(time_data, index=column_labels).T
    #     time_df.timestamp = time_df.timestamp.astype(str).str[:8].str.replace(":","-")

    for i, row in time_df.iterrows():
        cur.execute(time_table_insert, list(row))

    # load user table
    user_df = df[["userId", "firstName", "lastName", "gender", "level"]].copy(deep=True)
    user_df.drop_duplicates(inplace=True)

    # insert user records
    for i, row in user_df.iterrows():
        cur.execute(user_table_insert, row)

    # insert songplay records
    for index, row in df.iterrows():

        # get songid and artistid from song and artist tables
        cur.execute(song_select, (row.song, row.artist, row.length))
        results = cur.fetchone()

        if results:
            songid, artistid = results
        else:
            songid, artistid = None, None

        # insert songplay record
        #         time = str(dt.datetime.fromtimestamp(row.ts/1000).time())[:8]
        time = str(dt.datetime.fromtimestamp(row.ts / 1000))[:19]

        songplay_data = (
            time,
            row.userId,
            row.level,
            songid,
            artistid,
            int(row.sessionId),
            row.location,
            row.userAgent,
        )
        cur.execute(songplay_table_insert, songplay_data)


def process_data(cur, conn, filepath, func):
    """
    Read in the data from the filepath.
    Then processes the data using the function provided.
    Args:
        cur - cursor object
        conn - connection
        filepath - str, data location
        func - function, to process data
    """
    # get all files matching extension from directory
    all_files = []
    for root, dirs, files in os.walk(filepath):
        files = glob.glob(os.path.join(root, "*.json"))
        for f in files:
            all_files.append(os.path.abspath(f))

    # get total number of files found
    num_files = len(all_files)
    print("{} files found in {}".format(num_files, filepath))

    # iterate over files and process
    for i, datafile in enumerate(all_files, 1):
        func(cur, datafile)
        conn.commit()
        print("{}/{} files processed.".format(i, num_files))


def main():
    """
    Create the connection and cursor object.
    Orchestrate the data engineering.
    Then processes reads the song and log data and subsequently processes them.
    """
    conn = psycopg2.connect(
        "host=127.0.0.1 dbname=sparkifydb user=student password=student"
    )
    cur = conn.cursor()
    #     create_tables()

    process_data(cur, conn, filepath="data/song_data", func=process_song_file)
    process_data(cur, conn, filepath="data/log_data", func=process_log_file)

    conn.close()


if __name__ == "__main__":
    main()
