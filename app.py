from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import shutil
import logging
from pathlib import Path

from audio_command_detector import AudioCommandDetector

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
    "https://audioclassify-56pt8nazw-tuananhworks-projects.vercel.app",  # Preview URL
    "https://audioclassifyfe.vercel.app",  # Production URL
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
    try:
        # Tạo file tạm một cách an toàn
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(audio_file.file, tmp)
            temp_file_path = tmp.name

        # Kiểm tra độ dài tối thiểu (1.5s)
        if os.path.getsize(temp_file_path) < 16000 * 2 * 1.5:
            raise ValueError("File âm thanh quá ngắn. Yêu cầu tối thiểu 1.5 giây.")

        # Gọi hàm dự đoán
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
        # Xóa file tạm
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.get("/health")
async def health_check():
    """
    Kiểm tra trạng thái server
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 