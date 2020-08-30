# Disaster Response Pipelines 
## There are 3 significant aspects of this project is as follows:
1) ETL Pipeline
2) Machine Learning Pipeline
3) Creating Web-App 

The RandomForestClassifier produced these results:
-Average precision: 0.90828547402
-Average recall: 0.928911097549
-Average f_score: 0.907891040215


# File Structure

## Data folder contains the following:

categories.csv: contains the disaster categories csv file
messages.csv: contains the disaster messages csv file
DisasterResponse.db: contains the emergency db which is a merge of categories and messages by ID
process_data.py: contains the scripts to run etl pipeline for cleaning data

## Model folder contains the following:

ML_Model.pickle: contains the RandomForestClassifier fit pickle file
ML_Results.pickle: contains the RandomForestClassifier model's result pickle file
classifier.pickle: contains the Grid Search's result pickle file
train_classifier.py: script to train the data

## App folder contains the following:

Run.py: Defines the app routes

# Required Libraries

nltk, sklearn, sqlalchemy, pandas, numpy, pickle
