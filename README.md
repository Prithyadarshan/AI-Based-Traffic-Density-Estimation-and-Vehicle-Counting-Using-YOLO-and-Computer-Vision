# AI-Based Traffic Density Estimation and Vehicle Counting Using YOLO and Computer Vision

An educational computer vision project for detecting vehicles, counting traffic, and estimating traffic levels from road images and videos. The application uses a YOLO object-detection model with OpenCV-based image/video processing and a Streamlit interface for uploading inputs and viewing results.

Given a traffic image or video, the application can identify supported vehicle categories, annotate detections, and present vehicle-count results. Video tracking, traffic-level classification, CSV export, and processed-video downloads depend on the features enabled in the local implementation.

> **Implementation note:** This README describes the intended project workflow. Confirm each feature against your actual `app.py` before presenting it as implemented. In particular, counting detections per frame is different from counting unique vehicles across a video, and vehicle count alone is not a calibrated measurement of road density.

## Project goals

This project explores how AI-based object detection and computer vision can support traffic monitoring. Its goals are to:

1. Load a traffic image or video supplied by the user.
2. Apply a YOLO model to detect supported vehicle classes.
3. Display bounding boxes and class labels on the input.
4. Calculate vehicle counts using the counting logic implemented in the application.
5. Estimate a traffic category using the configured method, if available.
6. Display annotated results through a Streamlit interface.
7. Export reports or processed media where those options are implemented.

The system provides visual estimates from the supplied footage. It does not, by itself, measure traffic flow, vehicle speed, road capacity, or calibrated road occupancy.

## What the project demonstrates

- Uploading traffic images and videos through a web interface.
- Running YOLO inference on visual traffic data.
- Using OpenCV to decode, process, and annotate image/video frames.
- Summarizing detected vehicle categories.
- Visualizing detection results with bounding boxes and labels.
- Processing video frame by frame.
- Optionally tracking vehicles across frames, if tracking is configured.
- Optionally assigning traffic categories using configurable thresholds.
- Optionally generating CSV reports and downloadable annotated videos.

## Model and detection approach

YOLO (You Only Look Once) is a family of deep-learning-based object-detection models. It predicts object classes and bounding boxes in an image in a single inference pipeline. In this project, YOLO is used as the vehicle-detection component; OpenCV supports image and video handling, and Streamlit provides the interactive interface.

The application may use a pretrained Ultralytics YOLO checkpoint such as `yolov8n.pt` if that is the model configured in `app.py`. Check the actual model-loading code before specifying a particular YOLO version or checkpoint in a report.

### Detection pipeline

1. Read an uploaded image or video.
2. Prepare the image/frame for model inference.
3. Run the configured YOLO model.
4. Filter detections by the vehicle classes and confidence threshold configured in the application.
5. Draw bounding boxes and class labels.
6. Calculate the counts supported by the application's counting logic.
7. Display and/or save the resulting annotations and summaries.

Detection quality depends on the checkpoint, confidence threshold, input resolution, camera position, lighting, occlusion, and the visual characteristics of the traffic scene.

## Vehicle classes

The supported classes depend on the model's class labels and the class filter used by the application. A common COCO-pretrained YOLO model includes these vehicle categories:

| Example category | Typical use |
|---|---|
| Car | Detect and count cars |
| Motorcycle | Detect and count motorcycles |
| Bus | Detect and count buses |
| Truck | Detect and count trucks |

Do not assume that every road user is included in the final count. Confirm the class IDs and filtering logic in the source code. If additional classes or a custom-trained model are used, update this table accordingly.

## Vehicle counting and traffic-level estimation

### Vehicle counting

For a still image, the application can count the detections that pass its configured class and confidence filters. For video, the meaning of a count depends on the implementation:

- **Per-frame detections:** the number of detected vehicles in a particular frame.
- **Accumulated detections:** the sum of detections across multiple frames; the same vehicle may be counted repeatedly.
- **Unique tracked vehicles:** distinct track IDs counted over time, if a tracker and suitable counting logic are enabled.
- **Line-crossing count:** vehicles counted when they cross a defined virtual line, if implemented.

These quantities are not interchangeable. Describe the exact counting method used by your application when reporting results.

### Traffic-level estimation

If the application labels traffic as Low, Medium, or High, document the thresholds and the quantity on which those thresholds operate. For example, a threshold applied to the number of detections in a frame produces a simple count-based traffic category, not a validated traffic-engineering density measurement.

True traffic density is generally expressed as vehicles per unit distance or estimated from calibrated road regions and occupancy. A count-only category should therefore be described as an **estimated traffic level** unless the project implements and validates a more rigorous density method.

## Project files

The following is a suggested organization. Keep the file tree aligned with the files that actually exist in your project.

```text
AI_Based_Traffic_Density_Estimation_Project/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── yolov8n.pt                 # Optional if downloaded automatically
├── data/
│   ├── input/
│   │   ├── images/
│   │   └── videos/
│   └── output/
│       ├── annotated_images/
│       ├── processed_videos/
│       └── reports/
├── src/                           # Optional separated modules
├── tests/                         # Optional tests
├── screenshots/                   # Application evidence
└── docs/                           # Report, diagrams, and notes
```

If the current application is implemented in a single `app.py`, the `src/`, `tests/`, and other folders may not yet be present. Add them only when you organize code or store corresponding artifacts. The model checkpoint may be downloaded automatically by Ultralytics when the application first runs, depending on its configuration.

## Run the application

### Requirements

- A Python version supported by the installed dependencies.
- `pip`.
- Sufficient memory for the input image or video.
- Internet access on the first run if the model weights need to be downloaded.
- Git is optional and only needed for version control.

### 1. Open the project directory

If you downloaded the project as a ZIP, extract it and open a terminal in the folder containing `app.py`.

### 2. Create a virtual environment

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file should match the packages imported by your actual application. Common dependencies for this type of project include `ultralytics`, `opencv-python`, `streamlit`, and `pandas`, but only retain packages your code requires. Avoid installing both headless and GUI OpenCV packages unless your setup specifically requires it.

### 4. Start Streamlit

```bash
python -m streamlit run app.py
```

Open the local address printed in the terminal. Streamlit commonly uses:

```text
http://localhost:8501
```

### 5. Analyze an image or video

1. Select the supported input mode in the application.
2. Upload a traffic image or video.
3. Run the analysis using the interface controls.
4. Review bounding boxes, labels, and the available vehicle-count summary.
5. If supported, review the estimated traffic category.
6. If implemented, download the CSV report and processed video.

## Outputs

Depending on the implementation, outputs may include:

- An image or video annotated with detection boxes and class labels.
- Vehicle counts by supported class.
- A processed video.
- A CSV file containing count summaries or frame-level results.
- A traffic-level label based on configured thresholds.

Use actual screenshots and sample results from your own run. Do not include invented accuracy values or performance claims.

## Evaluation

A working interface confirms that the application runs, but it does not by itself establish detection accuracy. For a meaningful evaluation, use representative traffic footage and manually verified labels or counts.

Possible evaluation measures include:

- **Detection precision:** the proportion of predicted detections that are correct.
- **Detection recall:** the proportion of ground-truth vehicles that are detected.
- **mAP:** a standard object-detection metric, when evaluated against labeled data.
- **Counting error:** the difference between the predicted count and the manually verified count.
- **Processing speed:** frames per second or processing time, measured on documented hardware.

For counting, report the input clip, counting definition, ground-truth procedure, and error calculation. For traffic-level classification, compare the predicted category with a clearly defined reference label. Do not report these metrics unless you have actually measured them.

## Troubleshooting

- **Model download fails:** check internet access, model filename, and the installed Ultralytics package.
- **Module not found:** activate the correct environment and install the dependencies listed in `requirements.txt`.
- **No vehicles detected:** check the selected model, confidence threshold, image quality, supported classes, and whether the vehicles are visible.
- **Video cannot be opened:** try a supported video codec/container and verify that the file is not corrupted.
- **Processed video is empty or unreadable:** verify frame dimensions, video-writer initialization, codec availability, and output-file handling.
- **Repeated counts in video:** check whether the app counts each frame independently or uses unique track IDs/line crossing.
- **Slow processing:** try a smaller input resolution, shorter clip, or compatible GPU setup; document any change in settings.
- **Traffic category seems inaccurate:** inspect the configured thresholds and confirm whether the category is based on per-frame counts, accumulated detections, or another measure.
- **CSV or download option is missing:** confirm that the relevant export feature is implemented and that the output file is successfully generated.

## Limitations and responsible use

- Detection performance can vary with camera angle, lighting, weather, resolution, vehicle size, and occlusion.
- A pretrained model may not generalize equally well to every location or camera.
- Frame-by-frame detections can count the same vehicle repeatedly when summed over a video.
- Tracking can lose or reassign IDs, particularly with occlusion or crowded scenes.
- A simple count threshold is not a calibrated measure of traffic density or congestion.
- Results should be reviewed before being used to inform operational traffic decisions.
- Uploaded footage may contain identifiable people or vehicle registration plates. Handle and share it responsibly.

This project is intended for educational and demonstration purposes. It is not a substitute for a validated traffic-management system.

## Possible extensions

- Add line-crossing or region-based counting.
- Integrate a tracker such as ByteTrack if it is not already used.
- Estimate road occupancy using a calibrated region of interest.
- Analyze traffic trends across time windows.
- Add charts for vehicle counts by class and time.
- Evaluate against a labeled traffic dataset.
- Compare model checkpoints and confidence thresholds.
- Add real-time camera input or deployment to a suitable server.
- Provide configurable camera regions and counting lines.

## Main technologies

- Python
- YOLO / Ultralytics (according to the configured checkpoint)
- OpenCV
- Streamlit
- Pandas (if used for report generation)

## References

- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/) — for implementations that use tracking
- [OpenCV documentation](https://docs.opencv.org/)
- [Streamlit documentation](https://docs.streamlit.io/)

## Author

**PRITHYADARSHAN T**  
B.E. Computer Science and Engineering (AI & ML)  
Saranathan College of Engineering, Tiruchirappalli, Tamil Nadu, India

- GitHub: [github.com/prithyadarshan](https://github.com/prithyadarshan)
- Portfolio: [prithyadarshan-thiyagarajan.netlify.app](https://prithyadarshan-thiyagarajan.netlify.app)

---

*An educational project exploring AI-based vehicle detection, computer vision, vehicle counting, and traffic-level estimation.*
