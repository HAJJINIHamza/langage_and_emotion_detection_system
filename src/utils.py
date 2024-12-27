import os 
import sys
import numpy as np
import pandas as pd 
import dill
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import re
import string
import nltk
from nltk.tokenize import word_tokenize
import pickle

from src.exception import CustomException

def save_obj (file_path, obj):
    try:
        dir_path = os.path.dirname (file_path)

        os.makedirs (dir_path, exist_ok = True)

        with open ( file_path, 'wb' ) as file_obj:
            dill.dump (obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model ( test_inputs, test_labels, model ):
    try:
        #Let's compute the F1 score
        y_predicted_prob = model.predict ( test_inputs )
        y_predicted = np.argmax ( y_predicted_prob, axis = 1 )

        #Computing F1 score and accuracy
        F1_score = f1_score ( test_labels, y_predicted, average = 'macro')
        Accuracy_score = accuracy_score ( test_labels, y_predicted )

        #Precision and recall
        Precision = precision_score (test_labels, y_predicted, average = 'macro')
        Recall = recall_score (test_labels, y_predicted, average= 'macro')

        return  { 'F1_score': F1_score, 'Accuracy_score': Accuracy_score,
                'Precision': Precision, 'Recall': Recall }
    
    except Exception as e:
        raise CustomException (e, sys)
    

def save_accuracy_historical ( shot, train_size, add_layers, num_unit, 
                              num_epochs, unfreezed_layers, lr_rate, Accuracy_score, 
                              F1_score, Precision, Recall, history):
    try: 
        #Storing metrics values
        shot = shot+1
        accuracy_historical = {}
        accuracy_historical[f'iter{shot}'] = {'training size': train_size,
                                            'added layers': add_layers,
                                            'num unit': num_unit,
                                            'Unfreezed layers': unfreezed_layers,
                                            'epochs': num_epochs,
                                            'learning rate': lr_rate,
                                            'accuracy': Accuracy_score,
                                            'f1 score': F1_score,
                                            'precision': Precision,
                                            'recall': Recall,
                                            "loss": history.history['loss'],
                                             "val_loss": history.history["val_loss"] }

        #Storing the accuracy historincal in our local
        with open ( 'artifacts/accuracy_historical_arb.txt', 'a' ) as file:
            for key, value in accuracy_historical.items():
                file.write ( f'{key} : {value} \n' )
    except Exception as e:
        raise CustomException ( e, sys )
    

def load_object (file_path):
    try:
        with open ( file_path, "rb" ) as f:
            return dill.load(f)
        
    except Exception as e:
        raise CustomException(e, sys)


class TextPreprocessor():
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        # Encoding labels
        self.label_encoder.fit(y)
        return self

    def transform(self, X, y=None):
        try:
            X_processed = []
            for text in X:
                processed_text = self.preprocess_text(text)
                X_processed.append(processed_text)
            return X_processed
        except Exception as e:
            raise CustomException (e, sys)

    def preprocess_text(self, text):
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            # Remove links
            text = re.sub(r'http\S+', '', text)
            # Remove special characters (excluding numbers)
            text = re.sub(r'[!@#$(),\n"%^*?\:;~`\\\[\]]', ' ', text)
            text = re.sub('\s+', ' ', text)  # remove extra whitespace
            # Lowercasing
            text = text.lower()
            # Remove Punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))
            # Tokenization
            tokens = word_tokenize(text)
            return " ".join(tokens)
        
        except Exception as e:
            raise CustomException (e, sys)




