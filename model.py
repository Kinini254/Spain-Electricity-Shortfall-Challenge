"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    # ------------------------------------------------------------------------
    # set the first column to be the index
    feature_vector_df = feature_vector_df.set_index('Unnamed: 0')
    # convert the time variable to a datetime datatype
    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'])
    # extract relevant metrics like year, month, weekday, hour from the time feature
    feature_vector_df['Year'] = feature_vector_df['time'].dt.year
    # extract month
    feature_vector_df['Month'] = feature_vector_df['time'].dt.month
    # extract weekday
    feature_vector_df['Weekday'] = feature_vector_df['time'].dt.dayofweek
    # extract hour from the time feature
    feature_vector_df['Hour'] = feature_vector_df['time'].dt.hour
    # we now drop the time feature since the relevant metrics have been extracted
    feature_vector_df.drop("time", axis=1, inplace=True)
    # encode the Valencia_wind_deg values as numbers
    feature_vector_df['Valencia_wind_deg'] = feature_vector_df['Valencia_wind_deg'].str.extract(r'(\d+$)')
    # convert the data type to numeric
    feature_vector_df["Valencia_wind_deg"] = pd.to_numeric(feature_vector_df["Valencia_wind_deg"])
    # encode the Valencia_wind_deg values as numbers
    feature_vector_df['Seville_pressure'] = feature_vector_df['Seville_pressure'].str.extract(r'(\d+$)')
    # convert the data type to numeric
    feature_vector_df["Seville_pressure"] = pd.to_numeric(feature_vector_df["Seville_pressure"])
    # impute missing values using mode
    feature_vector_df["Valencia_pressure"].fillna(1016, inplace=True)
    predict_vector = feature_vector_df

    #return feature_vector_df
    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
