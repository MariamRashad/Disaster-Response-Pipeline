# import libraries
import sys
import os
import pandas as pd
import re
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''
    INPUT 
        database_filepath - Filepath used for importing the database     
    OUTPUT
        Returns the following variables:
        X - Returns the input features.  Specifically, this is returning the messages column from the dataset
        Y - Returns the categories of the dataset.  This will be used for classification based off of the input X
        y.keys - Just returning the columns of the Y columns
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df =  pd.read_sql_table('DisasterResponse', engine)
    X = df.message.values
    y = df.iloc[:,5:]
    return X, y, y.keys()


def tokenize(text):
    """ Normalize and tokenize
    """
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:

        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
        """
          Builds machine learning pipeline
        """
   pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'vect__min_df': [1, 5],
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [10, 25],
                'clf__estimator__min_samples_split': [2, 4]}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
   '''
    the Function required to evaluate the model and return the classification and accurancy score.
    Inputs: Model, X_test, y_test, Catgegory_names
    Outputs: Prints the Classification report & Accuracy Score
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))

def save_model(model, model_filepath):
    pass
   '''
    Save the model to a specified path
    
    Parameters:
    model: the ML model
    model_filepath: the file path that the model will be saved
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
      '''
    the Main function that train the classifier
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
    