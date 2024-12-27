import sys 
import os
import pandas as pd 
import pickle
from termcolor import colored
# Add the project root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
#from src.utils import load_object 
from src.exception import CustomException
from src.utils import TextPreprocessor


import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFXLMRobertaForSequenceClassification



class PredictPipeline:
    def __init__(self):
        pass
    
    def predict_emotion ( self, text: list ):
        try :
            #Detecting the text language
            #print ( "Detecting the text language" )
            #language_model_name = "papluca/xlm-roberta-base-language-detection"
            #language_detector = pipeline(task = "text-classification", model = language_model_name) 
            #language = language_detector ( text )[0]['label'] 
            #Importing language detector
            #Transforming text to a list in order to do prediction per batches
            if type(text) != list:
                text = [text]
            #Importing language detector
            with open ( "artifacts/experiments_models/MultinomialNB.pkl", "rb" ) as file:
                model = pickle.load ( file )

            #Idx to label
            idx_to_label = {0: "ar", 1: "dr_ar", 2: "dr_ltn", 3: "en", 4: "fr"} 
            idx = model.predict (text)
            print ( colored (f" Language of text: {text} is < {idx_to_label [idx.item()]} >", "yellow", "on_black" )) 
            language = idx_to_label [idx.item()]
            #num_to_emotion mapping
            num_to_emotion = {0: "happy", 1: "sad", 2: "angry", 3: "fear", 4: "surprised", 5: "neutral"}


            if language in ["ar"]:
                #Import model and tokenizer
                print ( colored (f" Importing arabic bert model \n", "yellow", "on_black") )
                model_path = "artifacts/experiments_models/arabic_bert_local"
                tokenizer_path = "artifacts/tokenizers/arabic_bert_tokenizer"
                arabic_model = tf.keras.models.load_model ( model_path, 
                                                           custom_objects = { 'TFAutoModelForSequenceClassification': 
                                                                             TFAutoModelForSequenceClassification } 
                                                            )
                #arbic_model = TFAutoModelForSequenceClassification.from_pretrained ( model_path )
                tokenizer = AutoTokenizer.from_pretrained ( tokenizer_path )
                
                #Tokenizing text
                max_length = 500
                tokenized_text = tokenizer ( text, padding = True, truncation = True,  return_tensors = "tf"  )
                tokenized_text = { 'input_ids': tokenized_text["input_ids"],
                                   'attention_mask': tokenized_text["attention_mask"] }
                #Model prediction
                predicted_probabilities = arabic_model.predict ( tokenized_text )
                emotions = num_to_emotion [predicted_probabilities.argmax (axis = 1 )[0]]
                print ( colored (f"emotion of text: {text} is << {emotions} >>", "yellow", "on_black") )
                return emotions
            
            elif language in ["fr", "en"]:
                #Import model and tokenizer
                print ( colored (f" Importing XLMRoberta model \n", "yellow", "on_black") )
                model_path = "artifacts/experiments_models/xlm_roberta_model"
                tokenizer_path = "artifacts/tokenizers/xlm_roberta_tokenizer"
                eng_fr_model = tf.keras.models.load_model ( model_path, 
                                                           custom_objects= {'TFXLMRobertaForSequenceClassification': 
                                                                            TFXLMRobertaForSequenceClassification} 
                                                           )
                #arbic_model = TFAutoModelForSequenceClassification.from_pretrained ( model_path )
                tokenizer = AutoTokenizer.from_pretrained ( tokenizer_path )
                
                #Tokenizing text
                max_length = None
                tokenized_text = tokenizer ( text, padding = True, truncation = True, return_tensors = "tf"  )
                tokenized_text = { 'input_ids': tokenized_text["input_ids"],
                                   'attention_mask': tokenized_text["attention_mask"] }
                #Model prediction
                predicted_probabilities = eng_fr_model.predict ( tokenized_text )
                emotions = num_to_emotion [predicted_probabilities.argmax (axis = 1 )[0] ]
                print ( "###############################################" )
                print ( colored (f"emotion of text: {text} is << {emotions} >>", "yellow", "on_black") )
                print ( "###############################################" )
                return emotions
            
            else:
                print ( "################################################################" )
                print ( colored (f"This language isn't supported yet. Only support: [ar, fr, en] ", "yellow", "on_black") )
                print ("##################################################################")

        except Exception as e:
            raise CustomException (e, sys)

#This class is benificial in case we want to deploy our model in an app using FastAPI for exemple
class CustomData:
    def __init__( self, text: str, text_language:str ):
        self.text = text
        self.text_language = text_language

    def get_data_as_data_frame (self):
        try:
            # Transforming our data to a dataframe
            data_dict = { "text": [self.text] }
            return pd.DataFrame ( data_dict )
        
        except Exception as e:
            raise CustomException (e, sys)


if __name__ == "__main__":   
    while True:
        print ( "Starting predictions" )
        #text = "شعرت بالحزن عندما تركت المدينة."       
        text = input ( colored ("""PLEASE ENTER YOUR TEXT HERE : >>  """, "black", "on_white") )
        if text:
            emotion = PredictPipeline().predict_emotion ( text = text )
        else:
            continue
        

 
