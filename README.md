## Project 2: Classifying Messages for Disaster Response
Code for pipelines for the ETL process and the creation of a machine learning model to allow the classification of messages about disasters in terms of their needed response. Produces a web app to visualise the traing set used to produce the model as well as allowing new messages to be classified. 

Table of Contents 
---

- [Installations](#installation)
- [Motivation](#motivation)
- [Files](#files)
- [Usage](#Usage)
- [Authors](#authors)
- [Acknowlegements](#acknowledgements) 

## Installation
This code has a number of required dependencies which have to be pre-installed to work. They include:

- numpy
- pandas
- ntlk
- sklearn
- sqlalchemy
- pickle

All python scripts include installation at beginning of all packages they require. The required datasets are also included within this repository - both "disaster_categories.csv" and "disaster_messages.csv"

## Motivation 
The motivation behind this project was to allow for the classification of messages around disasters such as war and extreme weather events in terms of the reponse they require. This ensures that the help people recieve in situation such as this is both relevant and timely. The creation of pipelines in order to do this ensures that the process can be repeated and new data classified easily. A web app allows both new classifcations to be made easily for a non-technical user as well as providing visualisations on the training set and any limitations it may provide. 

## Files 
Attached are folders of python files which allow for the cleaning of data, the building of the machine learning model and the code needed to run the web app. Files of note within it are: 

- "process_data.py" : code used to merge the two datasets and clean them. Stored within the data folder.
- "train_classifier" : code used to build a machine learning model. Stored within the models folder.
- "run.py" : the  code needed to run the web app. Stored within the app folder.

How to use these files is mentioned in the forthcoming [usage](#Usage)  section.

## Usage 
In order to be able to use the above files and produce the web app. A number of steps should be followed on a IDE. To run the ETL pipeline for cleaning the data you enter 
- "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db"
Then the followinbg command runs the Machine Learning model builder pipeline
- "python models/train_classifier.py data/DisasterResponse.db models/classifier"
The navigate to app directory and then run the run.py document to build a preview of the app.

## Authors

All code created by [Joshua Lindsay](https://github.com/josh-lindsay2023) following a Udacity template. 
Disaster Data taken from [Appen](https://appen.com/)

## Acknowledegments

Thanks to StackOverflow for making this analysis possible via their disaster data. Thanks to UDACITY for helping me develop my coding and data science knowledge to carry out this analysis. 
