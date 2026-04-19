import streamlit as st
import cv2
import tempfile
import torch
import numpy as np
from ultralytics import YOLO
import os

st.set_page_config(page_title="OHE Arc Detection Pipeline V5", layout="wide")

# 1. Hardware Acceleration
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

st.title("OHE Arc Detection V5 - Hybrid Vision Pipeline")
st.markdown("""
This application implements the exact 3-Stage Hybrid Computer Vision Architecture from the V5 scripts:
1. **Stage 1 (Tight ROI)**: YOLOv8 model inference with a tight Central Region of Interest (CROI) cyan mask.
2. **Stage 2 (Lenient Trigger)**: Triggered by large, bright gradient blobs. Opens the mask to only block the bottom 10%.
3. **Stage 3 (Auto-Arc)**: Triggered by >40% plasma coverage. 100% open mask, instantly flags as an arc.
""")
st.info(f"Detected optimal hardware device: **{get_device().upper()}**")

# 2. UI Elements
st.sidebar.header("Settings")
model_upload = st.sidebar.file_uploader("Upload YOLO Model (.pt)", type=["pt"])
video_upload = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.75, 0.05)
process_button = st.sidebar.button("Process Video")

if "processed_video" not in st.session_state:
    st.session_state.processed_video = None

if process_button:
    if model_upload is None or video_upload is None:
        st.error("Please upload both a model (.pt) and a video file before processing.")
    else:
        with st.spinner("Processing video through the Hybrid Pipeline..."):
            tmp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            tmp_model_file.write(model_upload.read())
            tmp_model_file.close()
            model_path = tmp_model_file.name

            tmp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_video_file.write(video_upload.read())
            tmp_video_file.close()
            video_path = tmp_video_file.name

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            
            try:
                model = YOLO(model_path)
                model.to(get_device())

                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_pixels = width * height
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # Initialize writer with full width but half height for side-by-side
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height // 2))

                # --- MASKS SETUP ---
                cyan_bg = np.full((height, width, 3), (255, 255, 0), dtype=np.uint8)

                # 1. STAGE 1: TIGHT MASK
                pt1, pt2, pt3, pt4 = [int(width * 0.15), int(height * 0.18)], [int(width * 0.88), int(height * 0.10)], [int(width * 0.76), int(height * 0.82)], [int(width * 0.04), int(height * 0.64)]
                tight_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(tight_mask, [np.array([pt1, pt2, pt3, pt4])], 0, 255, -1)
                inv_tight_mask = cv2.bitwise_not(tight_mask)

                # 2. STAGE 2: EXPANDED MASK
                exp_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(exp_mask, [np.array([[0,0], [width,0], [width,int(height*0.90)], [0,int(height*0.90)]])], 0, 255, -1)
                inv_exp_mask = cv2.bitwise_not(exp_mask)

                # 3. STAGE 3: ZERO MASK
                zero_mask = np.full((height, width), 255, dtype=np.uint8)
                inv_zero_mask = cv2.bitwise_not(zero_mask)

                lower_plasma = np.array([50, 0, 220]) 
                upper_plasma = np.array([130, 100, 255]) 
                FLASH_TRIPWIRE_AREA = 500

                stframe = st.empty()
                progress_bar = st.progress(0)
                
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    
                    plasma_mask = cv2.inRange(hsv, lower_plasma, upper_plasma)
                    plasma_ratio = cv2.countNonZero(plasma_mask) / total_pixels
                    
                    _, bright_mask = cv2.threshold(hsv[:,:,2], 240, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    stage, trigger_type, text_color = 1, "Stage 1 (Tight ROI)", (255, 255, 0)
                    
                    if plasma_ratio > 0.40:
                        stage, trigger_type, text_color = 3, f"Stage 3 (Auto-Arc: {plasma_ratio*100:.1f}%)", (255, 0, 255)
                    elif contours:
                        largest = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest) > FLASH_TRIPWIRE_AREA:
                            b_mask = np.zeros((height, width), dtype=np.uint8)
                            cv2.drawContours(b_mask, [largest], -1, 255, -1)
                            m_hue, m_sat = np.median(hsv[:,:,0][b_mask==255]), np.median(hsv[:,:,1][b_mask==255])
                            if (50 <= m_hue <= 140) or (m_sat < 80):
                                stage, trigger_type, text_color = 2, "Stage 2 (Lenient Trigger)", (0, 165, 255)

                    # Apply Mask
                    if stage == 3: 
                        ai_v = cv2.add(cv2.bitwise_and(frame, frame, mask=zero_mask), cv2.bitwise_and(cyan_bg, cyan_bg, mask=inv_zero_mask))
                    elif stage == 2: 
                        ai_v = cv2.add(cv2.bitwise_and(frame, frame, mask=exp_mask), cv2.bitwise_and(cyan_bg, cyan_bg, mask=inv_exp_mask))
                    else: 
                        ai_v = cv2.add(cv2.bitwise_and(frame, frame, mask=tight_mask), cv2.bitwise_and(cyan_bg, cyan_bg, mask=inv_tight_mask))
                        
                    results = model.predict(ai_v, conf=conf_threshold, verbose=False)
                    arc_boxes = [b for b in results[0].boxes if int(b.cls[0]) == 0]
                    
                    report_frame = frame.copy()
                    
                    if len(arc_boxes) > 0 or stage == 3:
                        for b in arc_boxes:
                            x1, y1, x2, y2 = b.xyxy[0].int().tolist()
                            cv2.rectangle(report_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                        cv2.putText(report_frame, trigger_type, (width - 400, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)

                    # Create a side-by-side view of the original and the AI input
                    # Resize to fit display nicely
                    report_resized = cv2.resize(report_frame, (width // 2, height // 2))
                    ai_v_resized = cv2.resize(ai_v, (width // 2, height // 2))
                    combined_frame = np.hstack((report_resized, ai_v_resized))
                    
                    out.write(combined_frame)
                    
                    # Display in Streamlit
                    combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(combined_frame_rgb, channels="RGB", use_container_width=True)
                    
                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()
                out.release()
                st.success("Hybrid processing complete!")
                st.session_state.processed_video = output_path

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                if os.path.exists(model_path):
                    os.remove(model_path)
                if os.path.exists(video_path):
                    os.remove(video_path)

if st.session_state.processed_video:
    with open(st.session_state.processed_video, "rb") as file:
        video_bytes = file.read()
    
    st.download_button(
        label="Download Annotated Video",
        data=video_bytes,
        file_name="annotated_video.mp4",
        mime="video/mp4"
    )
