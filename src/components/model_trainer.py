import os 
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import TFXLMRobertaForSequenceClassification, XLMRobertaTokenizer
import tensorflow as tf

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from src.utils import evaluate_model, save_accuracy_historical


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join ( "artifacts", "model" )

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig ()

    def initiate_model_trainer ( self, train_arr, test_arr ):
        try:
            logging.info( "Initiate model trainer" )
            model_name = "02shanky/finetuned-twitter-xlm-roberta-base-emotion"
            tokenizer = XLMRobertaTokenizer.from_pretrained (model_name)
            tf_model = TFXLMRobertaForSequenceClassification.from_pretrained (model_name, from_pt = True)
            logging.info ("Extracting tokenizer completed")

            #Tokenizing the dataset
            train_inputs = tokenizer(train_arr['text'].tolist(), padding=True, truncation=True, return_tensors="tf")
            test_inputs = tokenizer(test_arr['text'].tolist(), padding=True, truncation=True, return_tensors="tf")
            logging.info ( "Tokenizing data completed" )

            #Let's encode the emotion column [emotion <--> label]
            emotion_to_num = {"happy": 0, "sad": 1, "angry": 2, "fear": 3, "surprised": 4, "neutral": 0}
            train_labels = [emotion_to_num[emotion] for emotion in train_arr['emotion']]
            test_labels = [emotion_to_num[emotion] for emotion in test_arr['emotion']]

            # Convert input data and labels to TensorFlow tensors
            train_inputs = {'input_ids': tf.convert_to_tensor(train_inputs['input_ids']),
                            'attention_mask': tf.convert_to_tensor(train_inputs['attention_mask'])}
            test_inputs = {'input_ids': tf.convert_to_tensor(test_inputs['input_ids']),
                            'attention_mask': tf.convert_to_tensor(test_inputs['attention_mask'])}
            train_labels = tf.convert_to_tensor(train_labels)
            test_labels = tf.convert_to_tensor(test_labels)

            logging.info ( "Tokenizing data - emotion encoding - transforming inputs&labels into tensors is completed" )

            #Buliding the model
            MODEL_PATH = ""
            if MODEL_PATH:
                logging.info ( "Importing an existing model" )
                #Fine tuning an existing model
                final_model = tf.keras.models.load_model ( MODEL_PATH, custom_objects={'TFXLMRobertaForSequenceClassification': TFXLMRobertaForSequenceClassification} )
                #Freezing layers
                final_model.layers[2].layers[0].trainable = False
                logging.info ( "Importing the existing model is completed" )

            else:
                logging.info ( "Building model from scratch" )
                #Building model 
                tf_model.layers[0].trainable = False
                #Defining a new classificatoin layer to be added to the end of the model
                #num_emotions = fr_train_dataset['emotion'].value_counts ().count()
                num_emotions = 5
                num_unit_1 = 256
                layer_1 = tf.keras.layers.Dense ( num_unit_1, activation = "relu", name = 'fully_connected_layer1' )
                #layer_2 = tf.keras.layers.Dense (num_unit_2, activation = "relu", name = 'fully_connected_layer2')
                classification_layer = tf.keras.layers.Dense ( num_emotions , activation='softmax', name='classification_layer' )

                #Inputs and Attention mask layers
                inputs = tf.keras.Input (shape =(None,), dtype=tf.int32, name = 'input_ids')
                attention_mask = tf.keras.Input (shape= (None,), dtype=tf.int32, name = 'attention_mask')

                #inputs = tf.convert_to_tensor(inputs)
                #attention_mask = tf.convert_to_tensor(attention_mask)

                # Pass the input through the pre-trained model layers
                #tf_model
                outputs = tf_model ({'input_ids': inputs, 'attention_mask': attention_mask})

                #Extracting the pooled output
                pooled_output = outputs ['logits']

                #Passing the model output through the classification layer
                classification_output = classification_layer ( layer_1 (pooled_output) )

                #Building the final model
                final_model = tf.keras.Model ( inputs = [inputs, attention_mask], outputs = classification_output )
                logging.info ( "model building completed" )
            
            #Compile the model
            lr_rate = 2e-4
            optimizer = tf.keras.optimizers.Adam ( learning_rate = lr_rate )
            loss = tf.keras.losses.SparseCategoricalCrossentropy ()
            metrics = ['accuracy']
            final_model.compile ( optimizer = optimizer, loss= loss, metrics= metrics )
            logging.info ( "Compling model completed" )

            print( final_model.summary () )

            #Training the model
            logging.info ( "Starting model fine tuning" )
            num_epochs = 1
            history = final_model.fit(
                {'input_ids': train_inputs ['input_ids'], 'attention_mask': train_inputs['attention_mask']},
                train_labels,
                validation_data=({'input_ids': test_inputs['input_ids'], 'attention_mask': test_inputs['attention_mask']}, test_labels),
                batch_size=8,
                epochs= num_epochs
            )
            logging.info ( "model fine-tuning completed" )

            #Model evaluation 
            model_report:dict = evaluate_model (  test_inputs, 
                                                  test_labels, model = final_model )
            
            #Save accuracu historical on artifacts
            save_accuracy_historical ( shot = 2, 
                                       train_size = train_arr.shape[0], 
                                       add_layers = 0,
                                       num_unit = 0,
                                       num_epochs = num_epochs,
                                       unfreezed_layers= " 1: head_classification_layer ",
                                       lr_rate= lr_rate,
                                       Accuracy_score= model_report ['Accuracy_score'],
                                       F1_score= model_report['F1_score'],
                                       Precision= model_report ['Precision'],
                                       Recall = model_report ['Recall'],
                                       history=history
                                     )

            return model_report

        except Exception as e:
            raise CustomException(e, sys)