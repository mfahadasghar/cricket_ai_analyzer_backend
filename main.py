from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cricket AI Analyzer API", version="1.0.0")

# CORS configuration for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model paths
STUMP_MODEL_PATH = os.path.join(MODEL_DIR, "best_stumps.pt")
BALL_MODEL_PATH = os.path.join(MODEL_DIR, "best_ball.pt")

# Lazy import to avoid loading models at startup
_stump_detector = None
_ball_analyzer = None


def get_stump_detector():
    """Lazy load stump detector"""
    global _stump_detector
    if _stump_detector is None:
        from detection.stump_detector import StumpDetector
        _stump_detector = StumpDetector(STUMP_MODEL_PATH)
        logger.info("Stump detector loaded")
    return _stump_detector


def get_ball_analyzer():
    """Lazy load ball analyzer"""
    global _ball_analyzer
    if _ball_analyzer is None:
        from analysis.tracker import BallAnalyzer
        _ball_analyzer = BallAnalyzer(BALL_MODEL_PATH)
        logger.info("Ball analyzer loaded")
    return _ball_analyzer


@app.get("/")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "service": "Cricket AI Analyzer",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Alternative health check endpoint"""
    return {"status": "ok"}


@app.post("/detect_stumps")
async def detect_stumps_endpoint(video: UploadFile = File(...)):
    """
    Detect stump positions in cricket video

    Returns:
        - file_id: Unique identifier for this video
        - stumps: JSON with stump coordinates
    """
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    json_path = os.path.join(OUTPUT_DIR, f"{file_id}_stumps.json")

    try:
        # Save uploaded video
        logger.info(f"Receiving video upload: {file_id}")
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        logger.info(f"Video saved: {video_path} ({len(content)} bytes)")

        # Run stump detection
        detector = get_stump_detector()
        stump_data = detector.detect(video_path, json_path)

        logger.info(f"Stump detection complete: {file_id}")

        # Extract 4 corner points for Flutter app
        image_pts = stump_data.get("image_points", {})
        stump_points = [
            image_pts.get("bowler_bottom_left"),
            image_pts.get("bowler_bottom_right"),
            image_pts.get("batsman_bottom_right"),
            image_pts.get("batsman_bottom_left"),
        ]

        return JSONResponse(content={
            "file_id": file_id,
            "stumps": stump_points,  # Send as list of [x, y] points
            "calibration": stump_data  # Also include full calibration for reference
        })

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in stump detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stump detection failed: {str(e)}")


@app.post("/analyze_ball")
async def analyze_ball_endpoint(
    video: UploadFile = File(...),
    stump_json: UploadFile = File(...)
):
    """
    Analyze ball trajectory given video and stump data

    Returns:
        - speed: Ball speed in km/h
        - line: Bowling line (Off/Middle/Leg)
        - length: Bowling length (Full/Good/Short/Yorker)
        - bounce: Bounce point coordinates
        - trajectory: List of ball positions
        - annotated_video_url: URL to download annotated video
    """
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    stump_path = os.path.join(UPLOAD_DIR, f"{file_id}_stumps.json")

    try:
        # Save uploaded files
        logger.info(f"Receiving files for analysis: {file_id}")
        with open(video_path, "wb") as f:
            f.write(await video.read())
        with open(stump_path, "wb") as f:
            stump_content = await stump_json.read()
            f.write(stump_content)

        # Validate and convert stump JSON
        try:
            stump_data = json.loads(stump_content)
            logger.info(f"Stump data loaded: {stump_data.keys() if isinstance(stump_data, dict) else 'invalid format'}")

            # Convert Flutter's simple format to full calibration format if needed
            if "stumps" in stump_data and "image_points" not in stump_data:
                # Flutter sends: {"stumps": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}
                # Convert to full format expected by BallAnalyzer
                points = stump_data["stumps"]
                if len(points) == 4:
                    # Get video dimensions
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    stump_data = {
                        "video": video_path,
                        "frame_index": 0,
                        "image_size": [frame_width, frame_height],
                        "image_points": {
                            "bowler_bottom_left": points[0],
                            "bowler_bottom_right": points[1],
                            "batsman_bottom_right": points[2],
                            "batsman_bottom_left": points[3],
                            "bowler_middle_stump": [(points[0][0] + points[1][0]) // 2, points[0][1]],
                            "batsman_middle_stump": [(points[2][0] + points[3][0]) // 2, points[2][1]]
                        }
                    }
                    # Rewrite the calibration file with full format
                    with open(stump_path, "w") as f:
                        json.dump(stump_data, f, indent=2)
                    logger.info("Converted simple stump format to full calibration format")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid stump JSON: {str(e)}")

        # Run ball analysis
        analyzer = get_ball_analyzer()
        results = analyzer.analyze(video_path, stump_path, OUTPUT_DIR, file_id)

        logger.info(f"Ball analysis complete: {file_id}")

        # Log results for debugging
        logger.info(f"Analysis results - Speed: {results.get('speed')}, Line: {results.get('line')}, Length: {results.get('length')}")

        # Create full URL for annotated video
        # Get the host from the request or use environment variable
        base_url = os.getenv("RENDER_EXTERNAL_URL", f"http://{HOST}:{PORT}")
        annotated_video_url = f"{base_url}/download/{file_id}_annotated.mp4"

        return {
            "speed": results.get("speed"),
            "line": results.get("line"),
            "length": results.get("length"),
            "bounce": results.get("bounce"),
            "trajectory": results.get("trajectory", []),
            "annotated_video": annotated_video_url,  # Full URL for Flutter to download
            "video_path": annotated_video_url,  # Alternative key for compatibility
            "file_id": file_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ball analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ball analysis failed: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download annotated video or JSON results
    """
    # Security: prevent path traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    if filename.endswith(".mp4"):
        media_type = "video/mp4"
    elif filename.endswith(".json"):
        media_type = "application/json"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)