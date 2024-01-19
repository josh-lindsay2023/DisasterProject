
### import all relevant projects######
import sys
import pandas as pd 
import re
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle 

###

def load_data(database_filepath):
    ''' function for the loading of the table from the sql database and any modifications which need to be made so that it an be used in future functions
    Inputs:
    database_filepath: where you can find the table you wish to use.
    Outputs:
    X : The message column which will serve as our input.
    Y: The different outputs (classications) which our model will be able to predict
    category_names: A list of the different outputs or categories available
    '''
    engine = create_engine("sqlite:///"+ database_filepath)
    df = pd.read_sql_table(database_filepath, "sqlite:///"+ database_filepath) #access the table from the required SQLite database
    
    df.dropna(inplace=True) # drop any missing values 
    X = df.message.values # set x to be the message column
    
    Y = df.drop(columns= ["id", "message","genre","original"]) #drops the columns which are not needed (not a possible category)
    category_names = Y.columns.tolist() #create a list of possible outputs
   
    
    return X,Y, category_names


def tokenize(text): 
    ''' function for tokenizing the text so that it can be used to in our model
    Inputs: 
    text - the piece of text which you want to manipulate
    outputs:
    clean_tokens - the tokens you have created via the tokenize process
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' #helps identify any urls in the text
    detected_urls= re.findall(url_regex, text) # uses the urlregex to find any urls in th text you input
    for url in detected_urls:
        text= text.replace(url, "placeholder") # replaces any url which is found with placeholder text
        
    tokens= word_tokenize(text) # tokenizes the input text into words
    lemmatizer= WordNetLemmatizer() #sets up lemmatizer function which will take words and give their roots
    clean_tokens= [] # empty array for word tokens
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() # lermmatizes word as well as converting it to lower case and removing any whitespace.
        clean_tokens.append(clean_tok) # appends these tokens into the empty list. 
  
    return clean_tokens


def build_model():
    ''' function to build the model which can then be fit
    outputs:
    cv- pipeline object which uses GridSearchcv as a way to iterate and find best parameters'''
    
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ]) # creates a pipeline to allow the building of the classifcation model 
    parameters= {
    
    'clf__estimator__n_estimators': [50, 100, 200],
    'clf__estimator__min_impurity_decrease':[0.0,0.25,0.5],
     'clf__estimator__max_features': [1,50,100]
    
    } # dictionary for possible values for different paramters within the model 
    cv = GridSearchCV(pipeline, param_grid=parameters)# goes through the combination of paramters for the pipeline and selects best
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' function to output the accuracy and other metrics for each of the different output categories of the model
    inputs:
    model = the fitted model which utlised the above pipeline.
    x_test= the unseen text input which you will use to test the model
    y_test = the classification for these unseen text messages
    
    outputs a report on each category and the model's performance
    '''
    y_pred= model.predict(X_test) #use model to predict on unseen data
    for category in range(len(category_names)):
        true_labels = Y_test.iloc[:, category]
        pred_labels = y_pred[:, category]
        report = classification_report(true_labels, pred_labels) # goes through each category and outputs a classiication report on each. 
        print(f"Category: {category_names[category]}")
        print(report)
        print("-----------------------------------") 


def save_model(model, model_filepath):
    ''' function for saving of model, if given model and model filepath'''
    pickle.dump(model, open(model_filepath, 'wb')) #saves model as pickle file


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)#performs a split to get a test and training set for the model
        
        print('Building model...')
        model = build_model()#builds model 
        
        print('Training model...')
        model.fit(X_train, Y_train)#fits the model on given data
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)# uses function to evlauate the model

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
