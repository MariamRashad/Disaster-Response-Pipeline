 # import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)
    df_temp_id = df['id']
    return df, df_temp_id


def clean_data(df):
    
     """
   the function Cleans 'df' Merged DataFrame, Refactors Variables (0 or 1), and Drops Duplicate Rows
        
    Returns:
        (DataFrame) df: Returns Cleaned DataFrame
    """
     categories = df['categories'].str.split(';', expand = True)
     row = categories.iloc[0]
     category_colnames = [x.split("-") for x in row]
     category_colnames = [item[0] for item in category_colnames]
     categories.columns = category_colnames
    
    # Converts Category Values to Either 0 or 1
     for column in categories:
          categories[column] = categories[column].str.split('-').str[1]
          categories[column] = pd.to_numeric(categories[column])

    # Drop The Original Categories Column From `df`
     df = df.drop(['categories'], axis=1)
    
    # Concatenate The Original DataFrame With The New `categories` DataFrame
     df = pd.concat([df, categories], axis=1)
    
    # Drops Duplicates
     df = df.drop_duplicates()
     return df
def save_data(df, database_filename):
    """
    saves our data to a database
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', con=engine, index=False, if_exists='replace')


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