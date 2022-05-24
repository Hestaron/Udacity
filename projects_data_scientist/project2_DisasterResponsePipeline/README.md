# DisasterResponsePipeline
The dashboard tool for NLP on tweets about disasters

![image](https://user-images.githubusercontent.com/87130574/145090866-31068199-438b-4bb4-bbf9-af5185193abc.png)
# What does the disaster dashboard do?
The dashboard analyzes tweets and decides wether these tweets do fit in one or multiple categories regarding disaster management. The categories light up green if the category is applicable for the tweet.
The tweets are classified using a CountVectorizer, tFidfTransformer and a MultiOutputClassifier using a RandomForestClassifier.


The disaster categories are:
```
['related', 'request', 'offer', 'aid_related', 'medical_help',
       'medical_products', 'search_and_rescue', 'security', 'military',
       'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
```

On the main page, the first graph shows a bar chart which shows the different message genres: news, direct and social.

The second graph displays the f1-score, recall and precision for the different categories.



# How can you use/execute the dashboard?
<ul>
       <li>To run ETL pipeline that cleans data and stores the date in the database
        <ul>
               <li>`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
              </ul>
    <li>To run ML pipeline that trains classifier and saves
           <ul>
                  <li>`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
           </ul>
   <li>Run the following command in the app's directory to run your web app.
          <ul>
                 <li>`python run.py`
          </ul>
</ul>

Where the text is "Enter a message to classify" one can input a tweet which will be classified.



# What files are in the repository?
<ul>
       <li>app
       <ul>
         <li>img
         <ul>
              <li>githublogo.png: GitHub logo
              <li>linkedinlogo.png: LinkedIn logo
         </ul>
         <li>templates
         <ul>
                <li>go.html: the html for the responses of the nlp model
                <li>master.html: the html base for the index page
                <li>run.py: the python file which starts the flask script that opens master.html
                </ul>
              </ul>
  <li>data
  <ul>
    <li>DisasterResponse.db: the database with information about categories and messages
    <li>categories.csv: the raw information about categories which is put into the database
    <li>message.csv: the raw message information which is put into the database
    <li>process_data.py: the python script which merges the categories and messages into the database
  </ul>
  <li>models
    <ul>
    <li>ML Pipeline Preparation.ipynb: sketch and quick testing
    <li>train_classifier.ipnyb: the classifier creation file, old and not in use anymore
    <li>train_classifier.py: The script that creates the model and saves the model and metrics as f1, recall and precision
    <li>train_model_functions.py: the functions used by train_classifier.py, splitted to be more readable
         </ul>
              </ul>

# Installations
Installation can be done by running the following from terminal ```pip install -r requirements.txt```.

# Acknowledgments
Data is provided by [FigureEight, now appen](https://appen.com/)
Project reviewed by [Udacity](https://www.udacity.com/)
