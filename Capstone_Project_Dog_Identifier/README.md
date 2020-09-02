# Disaster Response Pipelines 
## There are 3 significant aspects of this project is as follows:
1) ETL Pipeline
2) Machine Learning Pipeline
3) Creating Web-App 

The RandomForestClassifier produced these results:<br/>
Average precision: 0.90828547402<br/>
Average recall: 0.928911097549<br/>
Average f_score: 0.907891040215<br/>


# File Structure

## Data folder contains the following:

categories.csv: contains the disaster categories csv file<br/>
messages.csv: contains the disaster messages csv file<br/>
DisasterResponse.db: contains the emergency db which is a merge of categories and messages by ID<br/>
process_data.py: contains the scripts to run etl pipeline for cleaning data<br/>

## Model folder contains the following:

ML_Model.pickle: contains the RandomForestClassifier fit pickle file<br/>
ML_Results.pickle: contains the RandomForestClassifier model's result pickle file<br/>
classifier.pickle: contains the Grid Search's result pickle file<br/>
train_classifier.py: script to train the data<br/>

## App folder contains the following:

Run.py: Defines the app routes

# Required Libraries

nltk, sklearn, sqlalchemy, pandas, numpy, pickle
