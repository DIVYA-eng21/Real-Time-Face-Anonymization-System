Project Overview

This project implements a real-time face anonymization system using OpenCV and MediaPipe. The system detects faces from a live webcam feed and applies Gaussian blur to anonymize them, while displaying the real-time FPS.
It is designed as a privacy-preserving demonstration, showing hands-on skills in computer vision, real-time video processing, and MediaPipe Tasks API.

Features:

Real-Time Face Detection: Uses MediaPipe BlazeFace for high-performance face detection.
Adaptive Face Anonymization: Gaussian blur kernel scales based on face size for better anonymization.
FPS Overlay: Displays real-time frames per second for performance monitoring.
Robust Processing: Safe handling of bounding boxes near frame edges.
Lightweight and CPU-Friendly: Runs smoothly on a standard CPU webcam feed (~24–25 FPS).

Tech Stack

Python 3.x
OpenCV – for image processing, Gaussian blur, and webcam capture.
MediaPipe Tasks API – BlazeFace face detection with asynchronous live stream processing.
NumPy – for array manipulation and frame handling.

How It Works

Webcam captures frames in BGR format.
Frames are converted to RGB and wrapped in mp.Image.
BlazeFace detects faces asynchronously in LIVE_STREAM mode.
Detected faces’ bounding boxes are extracted and blurred using adaptive Gaussian blur.

Installation:
# Clone the repo
git clone https://github.com/<your-username>/face-anonymization.git
cd face-anonymization

# Install dependencies
pip install opencv-python mediapipe numpy

Note: You need the MediaPipe BlazeFace TFLite model (blaze_face_short_range.tflite) in the project directory.

Usage :
python face_anonymizer.py

The webcam window will open with faces blurred in real-time.
Press q to stop the application.




Real-time FPS is calculated and overlaid on the video.

Press q to exit.
