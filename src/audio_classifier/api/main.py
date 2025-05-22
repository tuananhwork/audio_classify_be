from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import shutil
import logging
from pathlib import Path

from audio_classifier.core.audio_command_detector import AudioCommandDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Command Detection API",
    description="API for detecting voice commands using machine learning",
    version="1.0.0"
)

# CORS configuration
origins = [
    "http://localhost:5173",  # Local development
    "https://frontend-2ipecti54-tuananhworks-projects.vercel.app",  # Preview URL
    "https://frontend-indol-six-79.vercel.app",  # Production URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = AudioCommandDetector()

@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    # Create a temporary file
    temp_file_path = f"temp_{audio_file.filename}"
    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Check file size (minimum 1.5 seconds)
        if os.path.getsize(temp_file_path) < 16000 * 2 * 1.5:  # 1.5s * 16kHz * 2 bytes
            raise ValueError("File âm thanh quá ngắn. Yêu cầu tối thiểu 1.5 giây.")
        
        # Get prediction
        result = detector.predict(temp_file_path)
        
        return {
            "status": "success",
            "data": {
                "predicted_class": result["data"]["predicted_class"],
                "confidence": float(result["data"]["confidence"]),
                "top3_predictions": result["data"]["top3_predictions"],
                "waveform": result["data"]["waveform"]
            }
        }
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass

@app.get("/health")
async def health_check():
    """
    Kiểm tra trạng thái server
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 