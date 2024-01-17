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
Jupyter notebook including the code used to run the analysis: ProjectCode.ipynb


## Usage
Download survey responses before use as stated in installation instructions. 
Code is commented and split into clear sections to allow ease of use. All parts should-be self explanatory. All contributions are welcome using the existing files (or added more recent surveys) to allow further analysis and insights to be given.

## Author

All code created by [Joshua Lindsay](https://github.com/josh-lindsay2023)
Survey Responses taken from [StackOverflow](https://insights.stackoverflow.com/survey)

## Acknowledegments

Thanks to StackOverflow for making this analysis possible via their surveys and responses. Thanks to UDACITY for helping me develop my coding and data science knowledge to carry out this analysis. 
