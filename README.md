# CHATBOT EMOTION DETECTION

Develop a model to detect emotions and implement it in a chatbot.

## Table of contents

1. [Introduction](#introduction)
2. [Project structure](#Project-structure)
3. [Getting Started](#getting-started)
4. [Checking the code](#checking-the-code) 
5. [test our system](#test-our-system)


## Introduction
The project "Chatbot Emotion Detection" is part of the larger Inwi Chatbot project. Its goal is to equip the chatbot with the ability to detect users' emotions from conversations and provide responses that correspond to these emotions. To achieve this, we developed a system based on AI models (transformers) that can detect emotions from text in four languages: English, French, Arabic, Darija-Arab, and Darija-Latin.

## Project structure
The project has the following structure:
```
chatbot_emotions_detection/
├── .gitignore
├── Dockerfile
├── Dockerfile.prediction
├── README.md
├── requirements.txt
├── artifacts
│   ├── experiements_models/
│   │   ├── arabic_bert_local
│   │   ├── xlm_roberta_model
│   │   ├── MultinomialNB.pkl
│   ├── tokenziers/
│   │   ├── arabic_bert_tokenizer
│   │   ├── xlm_roberta_tokenizer
│   ├── images/
│   │   ├── prediction pipeline.png
├── notebooks/
│   ├── 01_experiments.ipynb
│   ├── 02_experiments.ipynb
│   ├── 03_experiments.ipynb
│   ├── importing_and_preparing_arabic_dataset.ipynb
│   ├── importing_data.ipynb
│   ├── preparing_final_data.ipynb
├── src/
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
│   ├── __init__.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── __init__.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   ├── train_pipeline.py
│   │   └── __init.__.py
```

- ``Datasets\emotions_detection_datasets\02_final_data.xlsx`` We chose to concat all our datasets into one file ``02_final_data.xlsx``, this file is our final dataset after beeing cleaned, with 453708 rows it contains all the needed emotions in 5 languages : English, French, Arabic, Egyptian, Darija Arabic and Latin.  
- ``Artifacts``: contains project outputs like our models, tokenizers, some images, etc.
- ``notebooks`` : Contains all the jupyter notebooks where we conducted our experiments: Importing dataset, cleaning and preparing data, testing different models, fine-tuning them, etc.
- ``src``: Contains the files for project pipeline : Data ingestion and tranformation, training pipeline, etc.
- ``requirements.txt``: A list of dependencies for the project
- ``README.md``: The detailed documentation for the project
- ``Dockerfile``: The docker image for data_ingestion, data_transformation and model_trainer
- ``Dockerfile.prediction``: The docker image for our system

## Checking the code

The code is dockerized, if you want to test the code (Data ingestion and data transformation files), follow these steps: 

1. Pull the code 

2. Go to a terminal and access the project directory 
```
cd chatbot_emotions_detection
```
3. Build the docker image : 
```
docker build -t dockerimagename
```
4. After the image is succefully builded run it using : 
```
docker run dockerimagename
```
At the end of the execution process you should see something like : 

`` train_set size: (12777, 2)
test_set size: (1044, 2)
Data Transformation Completed ``

!!! If you want to change the language on wich the model will be fine-tunned go to ``data_ingestion.py`` file in src folder, than change the ``LANGUAGE`` variable in line 39 (available choices : "eng",  "egy", "arb-trans", "arb", "fr", "ar", "alg-latin", "arb_classic"): 

```            
    logging.info ( "Selecting language on wich model will be fine-tunned" )
    LANGUAGE = ["arb", "egy"]
    df = df [df['language'].isin (LANGUAGE)]
```

## Test our system 
System overview : 

![system overview](artifacts/images/Final%20system.png)

To test our system and models, please follow the instructions bellow: 

1. Clone the reprository :
```
git clone https://scm-platform.inwi.ma/inwi/data/data_ds_lab/chatbot_emotions_detection.git
```
3. Pull the code
2. Go to terminal and access the project directory
```
cd chatbot_emotions_detection
```
3. Buil docker image 
```
docker build -f docker.prediction -t dockerimagename
```
5. Run the application 
```
docker run -it dockerimagename
```
6. Once you run it, the app will ask you to provide a text, on witch it will run the inference, text can be in 3 langauges : English, French or Classical Arabic, if the text is in another langauge, the system will affirm that langauge is not yet supported. 