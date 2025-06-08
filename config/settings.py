import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration settings
class Config:
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
    BASE_FOLDER = os.getenv('BASE_FOLDER')
    FACE_DETECTION_MODEL = os.getenv('FACE_DETECTION_MODEL')
    FACE_RECOGNITION_MODEL = os.getenv('FACE_RECOGNITION_MODEL')
    FACE_IMAGE_SIZE = int(os.getenv('FACE_IMAGE_SIZE'))
    EMBEDDINGS_PATH = os.getenv('EMBEDDINGS_PATH')
    MODEL_FOLDER = os.getenv('MODEL_FOLDER')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    # JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_TOKEN_LOCATION = ['headers']
    JWT_HEADER_NAME = 'Authorization'
    JWT_HEADER_TYPE = 'Bearer'

    DB_CONFIG = {
        "host": os.getenv("POSTGRES_HOST"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "database": os.getenv("POSTGRES_DB"),
        "port": os.getenv("POSTGRES_PORT")
    }