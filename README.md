# 🎭 Real-Time Face Blur using MediaPipe & OpenCV

A real-time face detection and face blurring system built using MediaPipe Tasks Vision API and OpenCV.  
The application captures live webcam feed, detects faces using a pre-trained BlazeFace model, and applies Gaussian blur dynamically while displaying live FPS on the screen.

---

## 🚀 Features

- Real-time face detection
- Automatic face blurring using Gaussian Blur
- Dynamic blur intensity based on face size
- Live FPS counter overlay
- Asynchronous detection using LIVE_STREAM mode
- Safe bounding box handling (prevents frame overflow)
- Press `q` to exit safely

---

## 🧠 Project Overview

This project demonstrates real-time computer vision using MediaPipe’s modern Tasks API.

### Detection Pipeline

1. Webcam captures frame using OpenCV.
2. Frame is converted from BGR → RGB.
3. Frame is wrapped into MediaPipe Image format.
4. `detect_async()` sends frame with timestamp.
5. MediaPipe processes frame in LIVE_STREAM mode.
6. Callback function receives detection results.
7. Faces are blurred and displayed with FPS overlay.

---

## 🔎 Model Used

**BlazeFace Short Range (TFLite)**

- File: `blaze_face_short_range.tflite`
- Optimized for fast real-time face detection
- Lightweight and CPU-friendly
- Designed for short-range webcam use

Loaded using:

BaseOptions(model_asset_path=MODEL_PATH)

---

## ⚙️ LIVE_STREAM Mode

The detector runs in:

VisionRunningMode.LIVE_STREAM

This mode:

- Requires timestamps for each frame
- Runs asynchronously
- Uses a callback function (`result_callback`)
- Ideal for real-time video processing

---

## 🖼️ Face Blurring Logic

For each detected face:

face = input_array[y:y+h, x:x+w]  
k = max(15, (w+h)//20*2 + 1)  
face = cv2.GaussianBlur(face, (k, k), 0)

Why dynamic kernel size?

- Larger face → stronger blur
- Smaller face → lighter blur
- OpenCV requires odd kernel size for GaussianBlur

Bounding box is clipped to frame dimensions to avoid index errors.

---

## 🎯 FPS Calculation

FPS is computed every 1 second for stable measurement:

if current_time - prev_time >= 1.0:  
&nbsp;&nbsp;&nbsp;&nbsp;fps = frame_count / (current_time - prev_time)

Displayed using `cv2.putText()`.

---

## 📦 Requirements

Install dependencies:

pip install opencv-python mediapipe numpy

---

## 📁 Project Structure

face-blur-project/  
│  
├── face_detection.py  
├── blaze_face_short_range.tflite  
└── README.md  

---

## ▶️ How to Run

python face_detection.py  

Press:

q  →  Quit application

---

## ⚠️ Important Notes

- Webcam must be accessible.
- Keep the `.tflite` model file in the same directory.
- Good lighting improves detection accuracy.
- LIVE_STREAM mode requires proper timestamps.

---

## 🛠 Tech Stack

- Python
- OpenCV
- MediaPipe Tasks Vision API
- NumPy

---

## 💡 Future Improvements

- Face tracking instead of per-frame detection
- Toggle blur on/off with keyboard
- Add face recognition
- Deploy as web app (FastAPI + WebRTC)
- GPU acceleration
- Multi-face analytics (counting, logging)

---

## 👩‍💻 Author

Divya  
BTech | Electronics & Communication Engineering  
Interested in AI, ML, Deep Learning & Generative AI
