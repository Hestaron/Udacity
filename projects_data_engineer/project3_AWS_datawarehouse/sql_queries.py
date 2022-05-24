import configparser


# CONFIG
config = configparser.ConfigParser()
config.read('dwh.cfg')

# DROP TABLES
drop_exists = "DROP TABLE IF EXISTS "
staging_events_table_drop = drop_exists + "staging_events;"
staging_songs_table_drop = drop_exists + "staging_songs;"
songplay_table_drop = "DROP TABLE IF EXISTS songplay;"
user_table_drop = "DROP TABLE IF EXISTS users;"
song_table_drop = "DROP TABLE IF EXISTS songs;"
artist_table_drop = "DROP TABLE IF EXISTS artists;"
time_table_drop = "DROP TABLE IF EXISTS time;"

# CREATE STAGING TABLES

staging_events_table_create= ("""
CREATE TABLE IF NOT EXISTS staging_events
(
    artist VARCHAR,
    auth VARCHAR,
    firstName VARCHAR,
    gender CHAR(1),
    itemInSession INT,
    lastName VARCHAR,
    length NUMERIC,
    level VARCHAR,
    location VARCHAR,
    method VARCHAR,
    page VARCHAR,
    registration NUMERIC,
    sessionId INT,
    song VARCHAR,
    status INT,
    ts BIGINT,
    userAgent VARCHAR,
    userId INT
)
""")

staging_songs_table_create = ("""
CREATE TABLE IF NOT EXISTS staging_songs 
(
    num_songs INT,
    artist_id VARCHAR,
    artist_latitude NUMERIC,
    artist_longitude NUMERIC,
    artist_location VARCHAR,
    artist_name VARCHAR,
    song_id VARCHAR,
    title VARCHAR,
    duration NUMERIC,
    year INT
)
""")


# Create tables
 # Integer, varchar, sortkey, distkey
songplay_table_create = ("""
    CREATE TABLE songplay (
    "songplay_id" INT IDENTITY(0,1) PRIMARY KEY,
    "start_time" timestamp NULL,
    "user_id" INT NOT NULL DISTKEY,
    "level" VARCHAR NULL,
    "song_id" VARCHAR NOT NULL,
    "artist_id" VARCHAR NOT NULL,
    "session_id" INT NOT NULL,
    "location" VARCHAR NULL,  
    "user_agent" VARCHAR NULL);
""") # songplay_id SERIAL PRIMARY KEY, start_time timestamp, user_id int, level varchar, song_id varchar, artist_id varchar, session_id varchar, location varchar, user_agent varchar

user_table_create = ("""
    CREATE TABLE users (
    "user_id" INT PRIMARY KEY,
    "first_name" VARCHAR NULL,
    "last_name" VARCHAR NULL,
    "gender" VARCHAR NULL,
    "level" VARCHAR NULL)
""") # user_id pk, first_name varchar, last_name varchar, gender varchar, level varchar

song_table_create = ("""
    CREATE TABLE songs (
    "song_id" VARCHAR PRIMARY KEY,
    "title" VARCHAR NULL,
    "artist_id" VARCHAR NULL,
    "year" INT NULL,
    "duration" NUMERIC NULL)
""") # song_id pk, title varchar, artist_id varchar, year int, duration float

artist_table_create = (""" 
    CREATE TABLE artists (
    "artist_id" VARCHAR PRIMARY KEY,
    "name" VARCHAR NULL,
    "location" VARCHAR NULL,
    "latitude" NUMERIC NULL,
    "longitude" NUMERIC NULL)
""") # artist_id pk, name varchar, location varchar, lattitude float, longitude float

time_table_create = ("""
    CREATE TABLE time (
    "start_time" TIMESTAMP NOT NULL,
    "hour" int NOT NULL,
    "day" INT NOT NULL,
    "week" INT NOT NULL,
    "month" INT NOT NULL,
    "year" INT NOT NULL,
    "weekday" INT NOT NULL)
""") # start_time pk, hour int, day int, week int, month int, year int, weekday int


# STAGING TABLES
staging_events_copy = ("""
    copy staging_events from {}
    credentials 'aws_iam_role={}'
    format as json {}
    STATUPDATE ON
    compupdate on region 'us-west-2';
    """).format(config.get("S3","LOG_DATA"), config.get("DWH", "ARN").strip("'"), config.get("S3","LOG_JSON_PATH"))

staging_songs_copy = ("""
    copy staging_songs from {}
    credentials 'aws_iam_role={}'
    json 'auto'
    compupdate on region 'us-west-2';
    """).format(config.get("S3","SONG_DATA"), config.get("DWH", "ARN").strip("'"))


# INSERT TABLES
songplay_table_insert = ("""
    INSERT INTO songplay (             
        start_time,
        user_id,
        level,
        song_id,
        artist_id,
        session_id,
        location,
        user_agent
    )
                                        
    SELECT  se.ts/1000 AS start_time,
            se.userId                          AS user_id,
            se.level                           AS level,
            ss.song_id                         AS song_id,
            ss.artist_id                       AS artist_id,
            se.sessionId                       AS session_id,
            se.location                        AS location,
            se.userAgent                       AS user_agent
    FROM staging_events AS se
    JOIN staging_songs AS ss
        ON (se.artist = ss.artist_name)
    WHERE se.page = 'NextSong';
""")

user_table_insert = ("""
    INSERT INTO users (
        user_id,
        first_name,
        last_name,
        gender,
        level
    )
                                        
    SELECT  DISTINCT se.userId          AS user_id,
            se.firstName                AS first_name,
            se.lastName                 AS last_name,
            se.gender                   AS gender,
            se.level                    AS level
    FROM staging_events AS se
    WHERE se.page = 'NextSong';
""")

song_table_insert = ("""
    INSERT INTO songs (                 
        song_id,
        title,
        artist_id,
        year,
        duration
    )
                                        
    SELECT  DISTINCT ss.song_id         AS song_id,
            ss.title                    AS title,
            ss.artist_id                AS artist_id,
            ss.year                     AS year,
            ss.duration                 AS duration
    FROM staging_songs AS ss;
""")

artist_table_insert = ("""
    INSERT INTO artists (               
        artist_id,
        name,
        location,
        latitude,
        longitude
    )
                                        
    SELECT  DISTINCT ss.artist_id       AS artist_id,
            ss.artist_name              AS name,
            ss.artist_location          AS location,
            ss.artist_latitude          AS latitude,
            ss.artist_longitude         AS longitude
    FROM staging_songs AS ss;
""")

time_table_insert = ("""
    INSERT INTO time (                  
        start_time,
        hour,
        day,
        week,
        month,
        year,
        weekday
    )
                                        
    SELECT  se.ts/1000 AS start_time,
            EXTRACT(hour FROM start_time)      AS hour,
            EXTRACT(day FROM start_time)       AS day,
            EXTRACT(week FROM start_time)      AS week,
            EXTRACT(month FROM start_time)     AS month,
            EXTRACT(year FROM start_time)      AS year,
            EXTRACT(week FROM start_time)      AS weekday
    FROM    staging_events AS se
    WHERE se.page = 'NextSong';
""")

# QUERY LISTS

create_table_queries = [staging_events_table_create, staging_songs_table_create, songplay_table_create, user_table_create, song_table_create, artist_table_create, time_table_create]
drop_table_queries = [staging_events_table_drop, staging_songs_table_drop, songplay_table_drop, user_table_drop, song_table_drop, artist_table_drop, time_table_drop]
copy_table_queries = [staging_events_copy, staging_songs_copy]
insert_table_queries = [songplay_table_insert, user_table_insert, song_table_insert, artist_table_insert, time_table_insert]
