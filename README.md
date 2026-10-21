# YOLO-Based Traffic Density Estimation and Vehicle Counting

A Streamlit computer-vision demo that uses a pretrained YOLOv8 model to detect cars, motorcycles, buses, and trucks in traffic images and videos. Video processing uses ByteTrack to track detections across frames.

## Features
- Vehicle detection in images and videos
- Per-class vehicle counts
- Annotated image and processed-video output
- CSV report downloads
- Streamlit web interface

## Project structure
```text
yolo-traffic-density-estimation/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

> **Model weights:** The app uses `yolov8n.pt`. On first run, Ultralytics can download the pretrained weights if they are not present beside `app.py`. If you choose to commit the weights, check the applicable Ultralytics/YOLO model licensing terms first.

## Requirements
- Python 3.10+ (use a version supported by your installed Ultralytics/PyTorch packages)
- Internet access on first run if the model weights need to be downloaded

## Run on Windows
Open Command Prompt in this project folder, then run:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Run on macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## How it works
1. Upload a traffic image or video.
2. YOLOv8 detects the supported vehicle classes.
3. For videos, ByteTrack associates detections across frames.
4. The app displays annotations, counts, and downloadable CSV/video outputs.

## Important limitations
- The app uses a pretrained general-purpose model; performance depends on camera angle, image quality, lighting, and occlusion.
- Video counts represent unique tracked IDs across the clip, not instantaneous road occupancy. Track IDs may be missed or reassigned.
- The Low/Moderate/High categories use simple demonstration thresholds (5 and 15 vehicles). They are not validated congestion measurements. For meaningful traffic density estimation, calibrate against a representative dataset and consider road occupancy, region-of-interest counts, and time-based metrics.

## Future improvements
- Define road regions and count line crossings
- Estimate road occupancy and congestion over time
- Evaluate precision/recall on labeled traffic footage
- Add a dashboard for time-series traffic trends

## Author
**Prithyadarshan T**  
GitHub: [@prithyadarshan](https://github.com/prithyadarshan)
