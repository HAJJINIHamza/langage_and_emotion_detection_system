import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.utils import shuffle

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

#Data ingestion allows us to import data from its original source, split it into train and test sets then 
#store the data in artifacts.

@dataclass
class DataIngestionConfi:
    train_data_path: str= os.path.join('artifacts', 'train.csv')
    test_data_path: str= os.path.join ('artifacts', 'test.csv')
    raw_data_path: str= os.path.join ('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfi()

    def initiate_data_ingestion(self):
        logging.info ('Entered the data ingestion method or componenet')
        print ( os.getcwd() )
        try:
            DATA_PATH = 'Datasets/emotions_detection_datasets/02_final_data.xlsx'
            df = pd.read_excel (DATA_PATH)
            logging.info ( "Reading the dataset as dataframe" )
            os.makedirs (os.path.dirname (self.ingestion_config.train_data_path), exist_ok = True)
            
            logging.info ( "Selecting language on wich model will be fine-tunned" )
            LANGUAGE = ["arb", "egy"]
            df = df [df['language'].isin (LANGUAGE)]

            df.to_csv (self.ingestion_config.raw_data_path, index= False, header= True)

            logging.info ('Shuffling data and Train test split initiated')
            train_set, test_set = train_test_split (df, test_size = 0.075, shuffle = True, random_state = 42)

            print ( "train_set size:", train_set.shape )
            print ( "test_set size:", test_set.shape )

            train_set.to_csv ( self.ingestion_config.train_data_path, index = False, header = True )
            test_set.to_csv ( self.ingestion_config.test_data_path, index = False, header = True )

            logging.info ('Ingestion of data is completed')
            print ( "Data Ingestion completed" )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,

            )

        
        except Exception as e:
            raise CustomException (e, sys)


if __name__ == "__main__":
    #Data ingestion 
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    #Data Transformation
    data_transformation = DataTransformation ()
    train_arr, test_arr = data_transformation.initiate_data_transformation( train_data, test_data )    

    #Model Trainer
    #modelTrainer = ModelTrainer ()
    #model_report = modelTrainer.initiate_model_trainer(train_arr, test_arr)
    #print ( "Model report:", model_report )
    
