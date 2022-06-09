import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, from_unixtime, dayofweek


config = configparser.ConfigParser()
config.read('dl_prod.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config.get("CREDENTIALS", 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get("CREDENTIALS", 'AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """Creates the Spark Session"""
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark

def read_song_data(spark, input_data):
    """Read in the song data"""
    # get filepath to song data file
    song_data = f'{input_data}/song_data/*/*/*/*.json'
    # song_data = f'{input_data}/song_data/A/A/A/*.json'


    # read song data file
    return spark.read.json(song_data)

def process_song_data(spark, input_data, output_data):
    """Read the song data from input_data, process it, and output it to output_data"""
    # Read song data
    df = read_song_data(spark, input_data)

    # extract columns to create songs table
    songs_table = df.select(["song_id", "title", "artist_id", "year", "duration"]).drop_duplicates()

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data + 'songs.parquet', partitionBy=('year', 'artist_id'), mode='overwrite')

    # extract columns to create artists table
    artists_table = df.selectExpr(["artist_id", "artist_name as name", "artist_location as location", "artist_latitude as latitude", "artist_longitude as longitude"])

    # write artists table to parquet files
    artists_table.write.parquet(output_data + 'artists.parquet', partitionBy=("artist_id"), mode='overwrite')


def process_log_data(spark, input_data, output_data):
    """Read the log data from input_data, process it, and output it to output_data"""
    # get filepath to log data file
    log_data = f'{input_data}/log_data/*.json'

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.where(df.page=="NextSong")

    # extract columns for users table    
    user_table = df.selectExpr(["userId as user_id", "firstName as first_name", "lastName as last_name", "gender", "level"])
    
    # write users table to parquet files
    user_table.write.parquet(output_data + 'users.parquet', partitionBy=("user_id"), mode='overwrite')

    df = df.withColumn('start_time', from_unixtime(col('ts')/1000))
    time_table = df.select('start_time') \
        .withColumn("hour", hour("start_time")) \
        .withColumn("day", dayofmonth("start_time")) \
        .withColumn("month", month("start_time")) \
        .withColumn("week", weekofyear("start_time")) \
        .withColumn("year", year(df.start_time)) \
        .withColumn("weekday", dayofweek("start_time"))
    
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(output_data + "time.parquet", mode="overwrite")

    # read in song data to use for songplays table
    song_df = read_song_data(spark, input_data)

    # Create sql tables
    song_df.createOrReplaceTempView("song_table")
    df.createOrReplaceTempView("log_table")
    
    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql("""SELECT DISTINCT
    log.ts AS start_time,
    log.userId AS user_id,
    log.level,
    song.song_id,
    song.artist_id,
    log.sessionId AS session_id,
    location,
    userAgent AS user_agent
    FROM song_table as song
    LEFT JOIN log_table as log 
        ON log.artist = song.artist_name AND 
        log.song = song.title AND 
        song.duration = log.length
        """)

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(output_data + "songplays.parquet", mode="overwrite")


def main():
    """Processes the song and log data in spark and outputs those in parquet files."""
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://sparkify/"
    
    input_data = "data"
    output_data = "data"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
