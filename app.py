import os
import tempfile
from collections import Counter

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Traffic Analysis", page_icon="🚦", layout="wide")

st.title("🚦 YOLO-Based Traffic Density Estimation and Vehicle Counting")
st.write(
    "Upload a traffic image or video to detect vehicles, estimate counts, "
    "and view annotated results."
)

MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file yolov8n.pt was not found. Place it beside app.py "
            "or allow Ultralytics to download the pretrained model."
        )
    return YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8n.pt")

def traffic_level(count):
    # Demonstration thresholds only; calibrate with real local traffic data.
    if count <= 5:
        return "Low"
    if count <= 15:
        return "Moderate"
    return "High"

model = load_model()
uploaded = st.file_uploader(
    "Upload a traffic image or video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
)

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1].lower()
    is_video = suffix in {".mp4", ".avi", ".mov", ".mkv"}

    if not is_video:
        image = Image.open(uploaded).convert("RGB")
        st.subheader("Input image")
        st.image(image, use_container_width=True)

        if st.button("Analyze image", type="primary"):
            with st.spinner("Detecting vehicles..."):
                results = model.predict(image, verbose=False)
                result = results[0]
                counts = Counter()
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    if class_id in VEHICLE_CLASSES:
                        counts[VEHICLE_CLASSES[class_id]] += 1

                annotated = result.plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                total = sum(counts.values())

            st.subheader("Detection result")
            st.image(annotated_rgb, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Detected vehicles", total)
            c2.metric("Demo traffic level", traffic_level(total))

            report = pd.DataFrame(
                [{"Vehicle Type": name, "Count": counts.get(name, 0)}
                 for name in VEHICLE_CLASSES.values()]
            )
            st.dataframe(report, use_container_width=True)
            st.download_button(
                "Download CSV report",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="traffic_image_report.csv",
                mime="text/csv",
            )
    else:
        st.subheader("Input video")
        st.video(uploaded)

        if st.button("Analyze video", type="primary"):
            temp_input = None
            temp_output = None
            cap = None
            writer = None
            track_counts = Counter()
            seen_ids = set()
            frame_count = 0
            progress = st.progress(0, text="Preparing video...")
            status = st.empty()

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(uploaded.getbuffer())
                    temp_input = f.name

                cap = cv2.VideoCapture(temp_input)
                if not cap.isOpened():
                    st.error("Could not open this video. Try an MP4 or AVI file.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 25.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    temp_output = f.name
                writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )
                    result = results[0]
                    annotated = result.plot()
                    writer.write(annotated)
                    frame_count += 1

                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0].item())
                            if class_id not in VEHICLE_CLASSES:
                                continue
                            if box.id is not None:
                                track_id = int(box.id[0].item())
                                key = (class_id, track_id)
                                if key not in seen_ids:
                                    seen_ids.add(key)
                                    track_counts[VEHICLE_CLASSES[class_id]] += 1

                    if total_frames > 0:
                        progress.progress(
                            min(frame_count / total_frames, 1.0),
                            text=f"Processed {frame_count}/{total_frames} frames",
                        )
                    elif frame_count % 25 == 0:
                        status.write(f"Processed {frame_count} frames...")

                cap.release()
                cap = None
                writer.release()
                writer = None
                progress.progress(1.0, text="Video processing complete")

                st.subheader("Processed video")
                with open(temp_output, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                st.download_button(
                    "Download processed video",
                    data=video_bytes,
                    file_name="traffic_annotated.mp4",
                    mime="video/mp4",
                )

                total_unique = sum(track_counts.values())
                c1, c2 = st.columns(2)
                c1.metric("Unique tracked vehicles (whole clip)", total_unique)
                c2.metric("Demo traffic level", traffic_level(total_unique))

                report = pd.DataFrame(
                    [{"Vehicle Type": name, "Unique Tracked Count": track_counts.get(name, 0)}
                     for name in VEHICLE_CLASSES.values()]
                )
                st.dataframe(report, use_container_width=True)
                st.download_button(
                    "Download CSV report",
                    data=report.to_csv(index=False).encode("utf-8"),
                    file_name="traffic_video_report.csv",
                    mime="text/csv",
                )
                st.caption(
                    "Note: video counts are unique tracked IDs across the whole clip, "
                    "not the number of vehicles present at one moment. Tracking may miss "
                    "or reassign IDs. Traffic-level thresholds are for demonstration only."
                )
            finally:
                if cap is not None:
                    cap.release()
                if writer is not None:
                    writer.release()
                for path in (temp_input, temp_output):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
else:
    st.info("Upload an image or video to begin.")
