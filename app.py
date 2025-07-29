import sys
import os
import certifi
import pandas as pd
import pymongo

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME

# Load environment variables
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")

# MongoDB connection
ca = certifi.where()
client = pymongo.MongoClient(mongo_db_url, ssl=True, ssl_ca_certs=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# FastAPI App Setup
app = FastAPI()
templates = Jinja2Templates(directory="./templates")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Index Route
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

# Training Route
@app.get("/train")
async def train_route():
    try:
        logging.info("Training Pipeline Triggered")
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        logging.error(f"Training Failed: {e}")
        raise NetworkSecurityException(e, sys)

# Prediction Route
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        logging.info(f"Received file: {file.filename}")
        
        # Read CSV File
        content = await file.read()
        file.file.seek(0)  # Reset file pointer after reading
        df = pd.read_csv(file.file)
        logging.info(f"CSV Loaded Successfully. Shape: {df.shape}")
        
        # Load Preprocessor and Model
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=model)

        # Prediction
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred
        logging.info("Prediction Completed")

        # Save to CSV
        os.makedirs('prediction_output', exist_ok=True)
        df.to_csv('prediction_output/output.csv', index=False)

        # Render Table in HTML
        table_html = df.to_html(classes='table table-striped', index=False)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        logging.error(f"Prediction Failed: {e}")
        raise NetworkSecurityException(e, sys)

# Main Runner
if __name__ == "__main__": 
    app_run(app, host="0.0.0.0", port=8000)
