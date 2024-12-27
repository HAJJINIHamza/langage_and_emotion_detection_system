#Docker file for dockerizing data_ingestion, data_trainsformation.
FROM python 
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY Datasets/emotions_detection_datasets/02_final_data.xlsx ./Datasets/emotions_detection_datasets/02_final_data.xlsx
RUN mkdir -p ./logs
RUN mkdir -p ./artifacts
CMD [ "python", "./src/components/data_ingestion.py" ]

