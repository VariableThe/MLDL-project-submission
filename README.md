# OHE Arc Detection Pipeline (V5)

This project provides a Streamlit-based web interface for running real-time YOLO object detection inference on videos, specifically tailored for detecting electrical arcs on railway Overhead Equipment (OHE) and pantographs. 

This submission implements the **Version 5 (V5) 3-Stage Hybrid Computer Vision Architecture**, solving critical "Ghost" false positives and "Super Arc" false negatives using a combination of deep learning and OpenCV image processing.

## Pipeline Architecture

The application dynamically analyzes each frame to determine the best approach before passing it to the YOLOv8 model. The pipeline switches between three stages:

1. **Stage 1 (Tight ROI):** Used for standard frames. Applies a tight Central Region of Interest (CROI) cyan mask to block out regions where arcs never occur (e.g., the bottom and extreme sides of the frame), preventing "ghosts."
2. **Stage 2 (Lenient Trigger):** Triggered when OpenCV detects a massive, bright gradient blob (`FLASH_TRIPWIRE_AREA > 500`). The mask opens up (blocking only the bottom 10%) to allow YOLO to evaluate the massive flash.
3. **Stage 3 (Auto-Arc):** Triggered when a massive arc whites out the camera (>40% plasma coverage in the blue/white spectrum). The mask opens 100%, and the system automatically flags an arc without relying on YOLO.

## Features

- **Side-by-Side Visualization:** The Streamlit interface displays both the original frame (with bounding boxes) and the masked frame (what YOLO actually sees) side-by-side.
- **Hardware Acceleration:** Auto-detects and utilizes CUDA (NVIDIA) or MPS (Apple Silicon) if available, falling back to CPU.
- **Adjustable Confidence:** Real-time slider to adjust YOLO's confidence threshold.
- **Video Export:** Downloads the final side-by-side processed video as an MP4.

## Project Structure

- `app.py`: The main Streamlit application containing the UI, the 3-stage hybrid pipeline logic, video processing, and export capabilities.
- `models/`: Directory containing the V5 stable YOLO weights (`yolo_model.pt`).
- `data/`: Directory containing a sample video (`sample_video.avi`) for testing the model.
- `requirements.txt`: Python dependencies required to run the application.
- `CODE_EXPLANATION.md`: Detailed breakdown of the underlying mathematical logic and OpenCV processing.
- `.gitignore`: Configured to keep the repository safe for Git push.

## Prerequisites

- Python 3.8 or higher.
- A virtual environment is recommended.

## Setup Instructions

1. **Create and activate a virtual environment:**

   ```bash
   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure your virtual environment is activated.
2. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

3. Open the provided local URL in your web browser (usually `http://localhost:8501`).

## Usage

1. **Upload Model:** In the sidebar, upload the provided `.pt` model located in the `models/` directory.
2. **Upload Video:** In the sidebar, upload a video file. A test file is available in the `data/` directory.
3. **Adjust Confidence:** Use the slider to set the desired confidence threshold for YOLO predictions.
4. **Process:** Click **Process Video**. The application will run the Hybrid CV Pipeline and display the side-by-side progress.
5. **Download:** Once complete, use the **Download Annotated Video** button to save the MP4 file.
