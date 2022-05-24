
# Introduction
For the Udacity capstone project of the Nanodegree Data Scientist, data from Starbucks is analysed. We will see whether we can reliably predict whether we can predict if an customer will accept an offer.

## Table of contents


[Project Definition](#project_def)  
* [Project Overview](#project_overview)  
* [Problem Statement](#problem_statement)
* [Metrics](#metrics)

[Analysis](#analysis)  
* [Data Explorarion](#data_exploration)
<!--* [Data Visualization](#data_visualization)-->

[Methodology](#methodology)  
* [Data Preprocessing](#data_preprocessing)
* [Implementation](#implementation)
* [Refinement](#refinement)

[Results](#results)  
* [Model evaluation and validation](#model_eval)
* [Justification](#justification)

[Conclusion](#conclusion)  
* [Reflection](#reflection)
* [Improvement](#improvement)

[Deliverables](#deliverables)
* [Application](#application)


## Project Definition<a name="project_def"></a>

### Project overview<a name="project_overview"></a>
<!--Student provides a high-level overview of the project. Background information such as the problem domain, the project origin, and related data sets or input data is provided.--> Starbucks has data regarding customers, offers and transactions. They want to know if they can reliable predict whether a customer would complete an offer after viewing the deal. [Starbucks](https://www.starbucks.com/) provided the data for this project through [Udacity](https://www.udacity.com/).

#### The datasets

Offer portfolio: This dataset contains information regarding offer ids, offer rewards, communication channels, prerequisite, duration and offer type.  
![The portfolio data](images/offers.png)

Customer profile: his dataset contains gender, age and income from a customer, when they became a member and their id.  
![The customer profile](images/profile_raw.png)

Transcript data: This dataset containes when certain events took place. An event is an offer received, offer viewed, transacation and offer completed.  
![The transcript data](images/transcript_raw.png)


### Problem statement<a name="project_statement"></a>
<!--The problem which needs to be solved is clearly defined. A strategy for solving the problem, including discussion of the expected solution, has been made.-->
We will try to use the data provided by Starbucks to predict whether a certain offer would be accepted when provided at a certain time to a certain person. The three datasets provided by Starbucks wil be used for this. A classification machine learning model will be used for this case.

### Metrics<a name="metrics"></a>
<!--Metrics used to measure the performance of a model or result are clearly defined. Metrics are justified based on the characteristics of the problem.

For example, explain why you want to use the accuracy score and/or F-score to measure your model performance in a classification problem,-->
A classification algoritme is used and therefore the standard metrics would be accuracy, precision, recall, sensitivity and F-score. Since a false positive and a false negative are "equally bad" in this example. We won't focus on precision or recall. The data is not equally distributed in people that complete an offer or  not complete an offer. Therefore, the focus lies on F-score.
![Offer_completed](images/offer_completed.png)
## Analysis<a name="analysis"></a>

### Data Exploration and visualization<a name="data_exploration"></a>
<!--Features and calculated statistics relevant to the problem have been reported and discussed related to the dataset, and a thorough description of the input space or input data has been made. Abnormalities or characteristics about the data or input that need to be addressed have been identified.-->
When looking at the different characteristics of persons in the data. It is observed that there are slightly more males (8484) than females in this dataset (6129). There are some unknowns (2175) and others (212). But that is only a small part.  
Incomes are left skewed. In general, males earn less income and female a little more in this data. Others are neglectible in comparison to the amount of males and females and can be seen when viewing closely to the bottom.
![Income distributions for different genders](images/income_dist.png)

In the age categorie there are quite a few persons with the age 118. Since so many 118 year old persons would be quite an outlier. Especially those still able to come to Starbucks regularly. Therefore, where the age is 118, we assumed the age is unkown. Between 18 and 65 there are quite a few people in this sample. Then it steeply decreases. Also notice that there are quite a few unkowns.

![Age distribution in the dataset](images/age_dist.png)

Lastly when looking at the different offers in the dataset. It is seen that the offers are more or less equally present in the data.

![Offers offered to people](images/diff_orders.png)

Where the offer numbers can be found in this table:

![Offers](images/offers.png)

<!--### Data visualization<a name="data_visualization"></a>
Build data visualizations to further convey the information associated with your data exploration journey. Ensure that visualizations are appropriate for the data values you are plotting.-->

## Methodology<a name="methodology"></a>

### Preprocessing<a name="preprocessing"></a>
<!--All preprocessing steps have been clearly documented. Abnormalities or characteristics about the data or input that needed to be addressed have been corrected. If no data preprocessing is necessary, it has been clearly justified.-->
The different preprocessing steps in order to combine the data, clarify the data and handle outliers.

<ol>
<li>In the transcript data the "value" column contains json style data. The different json keys are "offer id", "offer_id" and "amount". These json values are turned to columns. Then "offer id" and "offer_id" are merged to one column.
![json data example](#images/json_data.png)
<li>Transcript data is joined to the profile data on offer_id.
<li>The portfolio data has a list in the column 'channels'. This list with channels where the offer is communicated is split in several columns (email, mobile, social, and web). In these columns there is a 1 if the offer was communicated through this channel and a 0 if it has not.
<li>The merged data is merged with the portfolio data to one dataframe.
<li>All time in the dataframe is transformed to hours.
<li>The data is split in 3 pieces. Offer received, offer viewed and ofer completed. Such that the data can be looked at 'per offer' instead of per 'action'
<li>Offer viewed and offer received are joined based on offer_id and person_id.
<li>A column is created to depict the time the offer ends. That is the time of the offer plus the duration of the offer.
<li>Offer completed is joined to the data on offer_id and person_id. But only if the offer is viewed. If the offer is not viewed but the offer is completed, this means that the person 'accidently' made a transaction without knowing there was an offer. Only the completed offers that were completed after the offer is viewed will be taken into account.
<li>As noticed above, in this data people are disproportionatally of the age 118. This is probably because of either the default value or either the maximum value. Since this is so unlikely to be true, we assume that people of the age 118 have an unknown age.
</ol>

### Feature engineering
After the data is ready to be 'used', still some processing needs to happen to prepare the data for machine learning processing.
<ol>
<li> The 'became_member_on' data can be transformed to months and years to take into account the data perspective. But a better way to look at the data could be to calculate how long people already are a member. Since this data is not until today, we calculate the data until the date that the latest person became a Starbucks member. Which is 2018-07-26.
<li>When we classify if a person completed an offer or not, we just have to see whether the person completed an offer after viewing it. Since the second is already fulfilled when creating the dataframe. A offer is completed when there is a 'time_offer_completed' present in the dataframe.
<li>Several floats that take integer values are transformed to integers. Just like several categorical values such as gender and offer_type are transformed to categoricals.
<li>The data is split in a training and test set (0.85/0.15).
<li>In the columns age and income there are missing values. Another column is created with True and False that indicates if those values were originally missing. In the original columns age and income, the missing values are filled with the medians from the training set.
</ol>

### Implementation<a name="implementation"></a>
<!--The process for which metrics, algorithms, and techniques were implemented with the given datasets or input data has been thoroughly documented. Complications that occurred during the coding process are discussed.-->
The data was first processed to make better sense of it. Then feature engineering was applied to make it better usable for the machine learning model.  

To explore what the best machine learning model would be the Python [package Pycaret](https://pycaret.org/) was used. This is a low-code machine learning library which is a great start to see what kind of models make the most sense to use for a problem. This is excluding hyper parameter tuning.

Random forests performed the best in this case. With slightly behind the random forest the lightgbm, this model did run significantly faster.
![Pycaret quick machine learning model comparison](images/pycaret_models.png)

For now we proceed with the Random forest model. Exploring the possibilities of the LightGBM model would be a good idea for future improvements. If we would've focussed on Recall as metric then Quadratic Discriminant Analysis would've seen an excellent model to explore.

### Refinement<a name="refinement"></a>
<!--The process of improving upon the algorithms and techniques used is clearly documented. Both the initial and final solutions are reported, along with intermediate solutions, if necessary.-->
#### Exploration
In the exploration phase with PyCaret we get with the scores beneath for the different metrics and the corresponding confusion matrix.
![PyCaret metric results](images/pycaret_rf.png)
![Pycaret confusion matrix](images/pycaret_conf_matrix.png)

#### Scikit-Learn base model
When using the scikit-learn RandomForestClassifier model we get these scores and corresponding confusion matrix.
![Scikit-Learn metric results](images/scikitlearn_metrics.png)
![Scikit-Learn confusion matrix](images/scikitlearn_cf_matrix.png)

#### Scikit-Learn tuned model
After tuning the scikit-learn model with a RandomizedSearchCV with the paramgrid beneath. These are the metrics and confusion matrix.
![Scikit-Learn tuned metric results](images/scikitlearn_tuned_metrics.png)
![Scikit-Learn tuned confusion matrix](images/scikitlearn_tuned_cf_matrix.png)
![Parameter grid hyper parameter tuning](images/param_grid.png)

## Results<a name="results"></a>
<!--
If a model is used, the following should hold: The final model’s qualities — such as parameters — are evaluated in detail.

Some type of analysis is used to validate the robustness of the model’s solution. For example, you can use cross-validation to find the best parameters.

Show and compare the results using different models, parameters, or techniques in tabular forms or charts.

Alternatively, a student may choose to answer questions with data visualizations or other means that don't involve machine learning if a different approach best helps them address their question(s) of interest.-->
The model's robustness is validated by RandomizedSearch Cross Validation. The parameters used in the final model are ![final parameters for the RandomForest](images/best_params.png)

Remember that we focus on the f1-score metric. When the the exploration, normal model and tuned model are compared we see this metrics:
![Metrics comparison](images/compare_metrics.png)

The tuned model is very slightly better than the exploration model and the not tuned model regarding the F1-score. This increase is really minimal in the metric. However, when we look at the confusion matrices it is evident that the tuned model is the best of the 3 models and that the exploration model did perform significantly worse in terms of many more false negatives and false positives.

## Conclusion<a name="conclusion"></a>

### Reflection<a name="reflection"></a>
<!--Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.-->
The project comprised of all the steps. Data exploration and cleaning really challenged myself on how to think about how to transform the data to a machine learning format. I was baffled at first on how I should approach modelling completing an offer without viewing an offer. In hindsight it seems obvious ofcourse.  

Then I used PyCaret to steer myself choosing a good starting point on what models would perform good and what would perform bad. I also looked at a regression model to see if chances could be modelled that an offer would be accepted. But those exploratory models performed bad, R<sup>2</sup> < 0.5. Therefore, I went for the classification approach.  

After the model was constructed I wanted to try Streamlit. I heard a lot about it and followed some webinars on it in the past. On top of that, it seemed easier to model an easy dashboard/interface in than Flask. Their good documentation online helped me a lot.

I learned a lot of new things such as how to approach this dataset and streamlit. So the project is already worth it.

### Improvement<a name="improvement"></a>
<!--Discussion is made as to how at least one aspect of the implementation could be improved. Potential solutions resulting from these improvements are considered and compared/contrasted to the current solution.-->
In every project there is a lot of room for improvement. A few examples are:
<ul>
<li>LightGBM model also performed really good. Just slightly worse than the Random Forest in the exploratory modelling. Therefore, looking into this approach could yield good results. Especially since I did not try hyper parameter tuning on the LightGBM model.</li>
<li>Another interesting way could be to see what the probabilities are that somebody would buy something at Starbucks without offer. This does not have to be done with a machine learning model but A/B testing also goes a far way in this case</li>
<li>The machine learning model is now trained on several pre-defined offers. It would be good to explore more if it is reasonable to expect that those pre-defined offers (with constant parameters) provide a good enough dataset to predict the single parameters such as reward or difficulty.
</ul>


## Deliverables<a name="deliverables"></a>

### Application<a name="application"></a>
A webapplication build is build. In order to run this first a model has to be created. 
<ol>
  <li>Create a separate environment should be created according to the environment.yml file. This can be achieved with running `conda env create -f environment.yml`.
  <li>Type `python run create_model.py` when the working directory is the repository root. To create the model, on my pc it took 35minutes to run this. If the untuned model is sufficient: Set `tuned=False` on line 325 in create_model.py.
  <li>Then type `streamlit run interface.py` and an interface should open. If not, follow the commands in the code editor or command window. Here you can try the model
 </ol>
 
The .ipynb file is a scratch notebook that is used to test, explore and generate images.

## Different sources
### Coding
https://www.statology.org/one-hot-encoding-in-python/  
https://stackoverflow.com/questions/56338847/how-to-give-column-names-after-one-hot-encoding-with-sklearn  
https://medium.com/nerd-for-tech/difference-fit-transform-and-fit-transform-method-in-scikit-learn-b0a4efcab804  
### Modelling
[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
### Parameter Tuning

https://towardsdatascience.com/gridsearch-vs-randomizedsearch-vs-bayesiansearch-cfa76de27c6b
https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
### Interpretation
https://towardsdatascience.com/accuracy-recall-precision-f-score-specificity-which-to-optimize-on-867d3f11124
https://www.statology.org/confusion-matrix-python/
