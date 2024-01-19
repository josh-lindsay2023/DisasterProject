
#### import packages################
import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
   ''' A function for loading the required datasets and merging them into 1 using the shared id column 
   Inputs: 
   messages_filepath: allows us to access the messages database
   categories_filepath : allows us to access the categories database
   Outputs:
   df: a merged dtabase of the two - on the id column
   '''

    #load two databases
    messages= pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    ''' Function for cleaning the merged database so it can be used in further analysis. Splits the category column into the seperate categories and drops any duplicates
    Inputs:
    df = newly merged database
    outputs:
    df= cleaned databse
    ''''
    
    categories= df["categories"].str.split(";", expand= True)#split the category column on each semi colon (new variable
    row = categories.iloc[0] #allows us to extract the category names
    category_colnames = row.apply(lambda x: x[:-2]) # removes the las two values of each category (the binary 1 or 0)
    categories.columns = category_colnames # sets the category column names
    for column in categories:
        categories[column] = categories[column].apply(lambda x: re.sub(r'\D', '', str(x)))#strips the category names from each of the values to just leave the binary number 
        categories[column] = categories[column].astype(int) # ensures the bianry column is a integer. 
    df.drop(columns=["categories"], inplace= True)# drop the original category column
   
    df = pd.concat([df, categories], axis=1) # merge the newly made categories df with the original one. 
    df= df.drop_duplicates() # drop any duplicates in the datset.
    return df

def save_data(df, database_filename):
    ''' a function for the saving of the dataframe which has been created as SQLite
    Inputs: 
    df= merged and cleaned database
    database_filename = what you want to call your new database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False) # saves the df in SQLite database 
    
  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()