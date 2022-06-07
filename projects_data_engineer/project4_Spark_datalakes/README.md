# Spark & Data Lakes

## Setting up the environment
Run the notebook ```spark_etl.ipynb```, if the error ```Exception: Java gateway process exited before sending the driver its port number while creating a Spark Session in Python``` is shown.
The culprit could be that no JAVA_HOME variable is set.

Check if java is installed by running ```where java``` in cmd. This should return a Path variable if Java is installed. If Java is not installed, install Java from [here](https://www.java.com/download/ie_manual.jsp). It should set the JAVA_HOME variable automatic.
If Java is installed or after Java is installed but the JAVA_HOME variable is not set. It can be set manually in ```edit the system environment variables``` in Windows OS.

