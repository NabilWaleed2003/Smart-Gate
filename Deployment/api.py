import os
import sys
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# ---------------------------------------------------------
# 1. Add the project root so Python can import Src
# ---------------------------------------------------------
# api.py lives in Deployment, so the parent folder must be on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

# Now the project modules can be imported safely
from Src.models.inference import TruckDetectorInference
from Src.models.tracker import TruckTracker

# ---------------------------------------------------------
# 2. Globals for the model and tracker
# ---------------------------------------------------------
detector = None
tracker = None

# Model path (relative to where the server is started)
MODEL_PATH = os.path.join(PROJECT_ROOT, "Weights", "model_tensorrt.engine")

# ---------------------------------------------------------
# 3. Application lifespan - load once
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts (loads the model into memory)
    and frees memory when the server shuts down.
    """
    global detector, tracker
    print("[INFO] Starting Smart Gate API...")
    print(f"[INFO] Loading TensorRT Model from: {MODEL_PATH}")
    
    try:
        # Initialize the model
        detector = TruckDetectorInference(MODEL_PATH, conf_threshold=0.45)
        
        # Initialize the tracker (assume the camera sends about 10 frames per second)
        tracker = TruckTracker(track_thresh=0.45, frame_rate=10)
        print("[INFO] Model and Tracker loaded successfully!")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise e
        
    yield  # Server is running and accepting requests
    
    # Cleanup when the server shuts down
    print("[INFO] Shutting down Smart Gate API... Clearing memory.")
    del detector
    del tracker

# ---------------------------------------------------------
# 4. FastAPI app
# ---------------------------------------------------------
app = FastAPI(
    title="Smart Gate AI API",
    description="API for real-time truck detection and gate control.",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------
# 5. Endpoints
# ---------------------------------------------------------

@app.get("/health")
async def health_check():
    """Simple health check to confirm the server is running."""
    return {"status": "ok", "message": "Smart Gate API is running smoothly!"}


@app.post("/api/v1/process_frame")
async def process_frame(file: UploadFile = File(...)):
    """
    Main endpoint: receive an image, analyze it, and return a gate decision.
    """
    # Ensure the uploaded file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # 1. Read the uploaded image as bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # 2. Decode to BGR (OpenCV format)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame_bgr is None:
            raise ValueError("Invalid image data")

        # 3. Pre-process for the model
        img_tensor = detector.process_frame(frame_bgr)
        
        # 4. Inference
        preds = detector.predict(img_tensor)
        
        # 5. Tracking
        tracked_objects = tracker.update(preds, frame_bgr)
        
        # 6. Gate decision
        open_gate = False
        detected_trucks_info = []

        if tracked_objects is not None and len(tracked_objects) > 0:
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id, conf, cls_id = obj
                track_id = int(track_id)
                
                # Include truck data in the response (useful for dashboards or logs)
                detected_trucks_info.append({
                    "track_id": track_id,
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

                # Open the gate only if this truck has not been counted before
                if tracker.is_new_truck(track_id):
                    open_gate = True
                    print(f"[GATE SIGNAL] NEW Truck Detected (ID: {track_id}). OPEN GATE!")

        # 7. Reply to the hardware / camera
        response = {
            "status": "success",
            "action": "open_gate" if open_gate else "keep_closed",
            "trucks_in_frame": len(detected_trucks_info),
            "details": detected_trucks_info
        }

        return JSONResponse(content=response)

    except Exception as e:
        print(f"[ERROR] Processing frame failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 6. Direct run for local testing
# ---------------------------------------------------------
if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
