import sys 
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException 
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join ('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object (self):
        """This function generate the preprocesser that do the data transformation for us."""
        try:
            #We don't need this function for now 
            pass
        except Exception as e:
            raise CustomException (e, sys)
            

    def initiate_data_transformation (self, train_path, test_path):
        try:
            train_df = pd.read_csv ( train_path )
            test_df = pd.read_csv (test_path)

            logging.info ( "Read train and test data completed" )

            logging.info ("Starting feature transformation")

            #preprocessing_obj = self.get_data_transformer_object ()

            #All observations with [s, lov, lo, j] as emotions to be deleted
            logging.info ( "Deleting meaningless values in emotions column" )
            train_idx_to_delete = train_df[train_df['emotion'].isin (['s', 'lov', 'lo', 'j'])].index
            train_df = train_df.drop (index= train_idx_to_delete)
            test_idx_to_delete = test_df[test_df['emotion'].isin (['s', 'lov', 'lo', 'j'])].index
            test_df = test_df.drop (index= test_idx_to_delete)

            #Doing the emotional mapping, emotion to their original form
            emotion_mapping = {'sadness':'sad', 'Sad': 'sad',
                                'joy': 'happy', 'happiness': 'happy', 'hapiness': 'happy', 'fun': 'happy', 'Laughing': 'happy', 'Happy': 'happy', 'love': 'happy', 'enthusiasm': 'happy', 'relief': 'happy', 'trust': 'happy',
                                'none': 'neutral', 'Neutral': 'neutral', 'boredom': 'neutral', 'empty': 'neutral', 'sympathy': 'neutral', 'Wondering': 'neutral', 'anticipation': 'neutral',
                                'Angry': 'angry', 'anger': 'angry', 'hate': 'angry', 'disgust': 'angry',
                                'Surprised': 'surprised', 'surprise': 'surprised',
                                'worry': 'fear', 'fear': 'fear'}
            
            #Remaining emotions : ['sad', 'happy', 'surprised', 'fear', 'angry', neutral]
            logging.info ( "Performing the emotion mapping to render every emotion to its original format" )
            train_df['emotion'] = train_df['emotion'].replace (emotion_mapping)
            test_df['emotion'] = test_df['emotion'].replace (emotion_mapping)


            train_df = train_df.drop ( columns = ['language'] )
            test_df = test_df.drop ( columns = ['language'] )
            logging.info ( "Deleted language column" )

            # Deleting rows with neutral as emotion
            index_to_delete = train_df[train_df["emotion"] == "neutral"].index
            train_df = train_df.drop ( index = index_to_delete )
            index_to_delete = test_df[test_df["emotion"] == "neutral"].index
            test_df = test_df.drop ( index = index_to_delete )
            logging.info ( "Deleted rows with neutral as emotion" )

            #Delete null values
            train_df.dropna(inplace = True)
            test_df.dropna(inplace = True)
            logging.info ( "Deleted null values" )
            
            #Delete duplicates 
            train_df.drop_duplicates (subset = "text")
            test_df.drop_duplicates  (subset = "text")
            logging.info ( "Deleted duplicates " )

            
            print ( "final  emotions :", train_df.emotion.value_counts() )
            print ( "final columns :", train_df.columns )
            print ( "train_set size:", train_df.shape )
            print ( "test_set size:", test_df.shape )

            logging.info ( f"final_columns{ train_df.columns }" )
            #train_df.columns = [text, emotion, language]
            logging.info ( "Data Transformation Completed" )
            print ( "Data Transformation Completed" )
            return (train_df, test_df)

           
        except Exception as e:
            raise CustomException (e, sys)

