from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uuid
from datetime import datetime
from google.cloud import storage
from google.cloud import bigquery
import tensorflow as tf
import numpy as np
import os
import json
import logging
from typing import Dict, Any, List
from google.api_core import exceptions as google_exceptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model API",
             description="Production ML Model Deployment",
             version="1.0.0")

class PredictionRequest(BaseModel):
    image: List[List[float]]
    model_version: str = "v1"

class PredictionResponse(BaseModel):
    prediction_id: str
    status: str
    result: Dict[str, Any] = None
    created_at: str

# Initialize GCP clients with error handling
def init_storage_client():
    try:
        return storage.Client()
    except Exception as e:
        logger.error(f"Failed to initialize Storage client: {str(e)}")
        return None

def init_bigquery_client():
    try:
        return bigquery.Client()
    except Exception as e:
        logger.error(f"Failed to initialize BigQuery client: {str(e)}")
        return None

storage_client = init_storage_client()
bigquery_client = init_bigquery_client()

# Storage configuration
ENABLE_STORAGE = os.getenv("ENABLE_STORAGE", "true").lower() == "true"
DATASET_NAME = os.getenv("BIGQUERY_DATASET", "model_prediction")
TABLE_NAME = os.getenv("BIGQUERY_TABLE", "prediction")

class ModelService:
    def __init__(self):
        self.models = {}
        self.storage_enabled = ENABLE_STORAGE and bigquery_client is not None
        self.load_models()
        if self.storage_enabled:
            self._init_storage()

    def _init_storage(self):
        """Initialize storage with error handling"""
        try:
            if bigquery_client:
                dataset_ref = bigquery_client.dataset(DATASET_NAME)
                table_ref = dataset_ref.table(TABLE_NAME)
                
                try:
                    bigquery_client.get_table(table_ref)
                    logger.info(f"BigQuery table {DATASET_NAME}.{TABLE_NAME} exists")
                except google_exceptions.NotFound:
                    logger.warning(f"Table {DATASET_NAME}.{TABLE_NAME} not found. Storage disabled.")
                    self.storage_enabled = False
                except Exception as e:
                    logger.error(f"Error checking BigQuery table: {str(e)}")
                    self.storage_enabled = False
        except Exception as e:
            logger.error(f"Error initializing storage: {str(e)}")
            self.storage_enabled = False

    def load_models(self):
        """Load models with fallback to local file"""
        try:
            if storage_client:
                # Try loading from GCS
                try:
                    bucket = storage_client.bucket(os.getenv("GCP_BUCKET_NAME"))
                    model_blob = bucket.blob("models/model_v1.h5")
                    model_blob.download_to_filename("/tmp/model.h5")
                    self.models["v1"] = tf.keras.models.load_model("/tmp/model.h5")
                    logger.info("Model loaded from GCS")
                except Exception as e:
                    logger.error(f"Error loading model from GCS: {str(e)}")
                    self._load_local_model()
            else:
                self._load_local_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError("Failed to load model")

    def _load_local_model(self):
        """Fallback to load model from local file"""
        try:
            local_model_path = os.getenv("LOCAL_MODEL_PATH", "model.h5")
            if os.path.exists(local_model_path):
                self.models["v1"] = tf.keras.models.load_model(local_model_path)
                logger.info("Model loaded from local file")
            else:
                raise FileNotFoundError(f"Model file not found at {local_model_path}")
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")
            raise

    def preprocess_input(self, image_data: List[List[float]]) -> np.ndarray:
        """Preprocess input data for model"""
        try:
            image = np.array(image_data)
            if len(image.shape) == 2:
                image = image.reshape(1, 28, 28, 1)
            elif len(image.shape) == 3:
                image = image.reshape(1, *image.shape)
            return image.astype('float32') / 255.0
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise ValueError(f"Invalid input format: {str(e)}")

    async def predict(self, image_data: List[List[float]], model_version: str) -> Dict[str, Any]:
        """Make prediction using the model"""
        try:
            model = self.models.get(model_version)
            if not model:
                raise ValueError(f"Model version {model_version} not found")
            
            processed_input = self.preprocess_input(image_data)
            prediction = model.predict(processed_input)
            
            return {
                "class_probabilities": prediction.tolist()[0],
                "predicted_class": np.argmax(prediction, axis=1).tolist()[0]
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

async def store_prediction(prediction_data: Dict[str, Any]):
    """Store prediction results in BigQuery if enabled"""
    if not (ENABLE_STORAGE and bigquery_client):
        logger.info("Storage disabled, skipping prediction storage")
        return

    try:
        row = {
            "prediction_id": prediction_data["prediction_id"],
            "model_version": prediction_data["model_version"],
            "input_shape": str(prediction_data["input_shape"]),
            "predicted_class": prediction_data["result"]["predicted_class"],
            "class_probabilities": prediction_data["result"]["class_probabilities"],
            "created_at": prediction_data["created_at"],
            "processing_time_ms": prediction_data.get("processing_time_ms", 0)
        }
        
        table_id = f"{DATASET_NAME}.{TABLE_NAME}"
        errors = bigquery_client.insert_rows_json(table_id, [row])
        
        if errors:
            logger.error(f"Error inserting rows: {errors}")
        else:
            logger.info(f"Successfully stored prediction {prediction_data['prediction_id']}")
    except Exception as e:
        logger.error(f"Error storing prediction: {str(e)}")

model_service = ModelService()

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    start_time = datetime.utcnow()
    try:
        prediction_id = str(uuid.uuid4())
        created_at = start_time.isoformat()
        
        result = await model_service.predict(
            request.image,
            request.model_version
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        prediction_data = {
            "prediction_id": prediction_id,
            "input_shape": np.array(request.image).shape,
            "result": result,
            "model_version": request.model_version,
            "created_at": created_at,
            "processing_time_ms": processing_time
        }
        
        if model_service.storage_enabled:
            background_tasks.add_task(store_prediction, prediction_data)
        
        return PredictionResponse(
            prediction_id=prediction_id,
            status="success",
            result=result,
            created_at=created_at
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    storage_status = "enabled" if model_service.storage_enabled else "disabled"
    return {
        "status": "healthy",
        "storage": storage_status,
        "model_versions": list(model_service.models.keys())
    }