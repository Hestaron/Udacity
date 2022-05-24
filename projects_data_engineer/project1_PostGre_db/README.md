# The Postgres song database

## Summary
This repository will read in the data about songs and logs in the data directory.
The 5 databases are songplay, users, songs, artists and time. The data from the songs is written to the songs and artists databases. The data from the logs is written tot he users and time databases.
Then those databases are combined to create the central database in the starformation called: 'songplay'.
This songplay database is the database optimized for queries. 

## Files
<ul>
    <li>create_tables.py - Deletes the tables if they exist and creates them anew.</li>
    <li>etl.ipynb - The notebook file used to create the etl.py file</li>
    <li>data - the data directory that contains the log and song directories</li>
    <li>etl.py - Creates the databases based on the data in the data directory</li>
    <li>sql_queries - contains the queries ran by etl.py and etl.ipynb</li>
    <li>test.ipynb - Used to check the inside of the databases</li>
</ul>

## Usage
Change directory to the workspace and run etl.py. 
Run ```python etl.py```

## The databases
<ul>
    <ul>songplay
        <li>songplay_id</li>
        <li>start_time</li>
        <li>user_id</li>
        <li>level</li>
        <li>song_id</li>
        <li>artist_id</li>
        <li>session_id</li>
        <li>location</li>
        <li>user_agent</li>
    </ul>
    <ul>users
        <li>user_id</li>
        <li>first_name</li>
        <li>last_name</li>
        <li>gender</li>
        <li>level</li>
    </ul>
    <ul>songs
        <li>song_id</li>
        <li>title</li>
        <li>artist_id</li>
        <li>year</li>
        <li>duration</li>
    </ul>
    <ul>
        <li>artists</li>
        <li>artist_id</li>
        <li>name</li>
        <li>location</li>
        <li>latitude</li>
        <li>longitude</li>
    </ul>
    <ul>time
        <li>start_time</li>
        <li>hour</li>
        <li>day</li>
        <li>week</li>
        <li>month</li>
        <li>year</li>
        <li>weekday</li>
    </ul>
</ul>
    
            
        
        
        
# Run the docker container
docker run -d –-name postgres -p 5432:5432 -e POSTGRES_PASSWORD=udacity -v postgres:/var/lib/postgresql/data postgres:14

