
import cv2
import mediapipe as mp
import time
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "blaze_face_short_range.tflite"


prev_time=time.time()
frame_count=0
fps=0
stop=False

def print_and_blur(result, input_image,timestamp_ns):
    global prev_time, frame_count,fps

    frame_count += 1
    current_time = time.time()

    if current_time - prev_time >= 1.0:
        fps = frame_count / (current_time - prev_time)
        prev_time = current_time
        frame_count = 0
    #fps calcuated.  
        
        
    global stop
    # Convert MediaPipe Image to NumPy array
    input_array = np.array(input_image.numpy_view(), dtype=np.uint8)

    if result.detections:
        for detection in result.detections:
            # Get bounding box
            bbox = detection.bounding_box
            x, y = int(bbox.origin_x), int(bbox.origin_y)
            w, h = int(bbox.width), int(bbox.height)
            
            # Apply blur using OpenCV
            h_img, w_img, _ = input_array.shape

            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)

            face = input_array[y:y+h, x:x+w]
            k = max(15, (w+h)//20*2 + 1)  # always odd
            face = cv2.GaussianBlur(face, (k, k), 0)
            input_array[y:y+h, x:x+w] = face

    
  
    #overlaying fps on webcam.
    cv2.putText(
    input_array,
    f"FPS: {int(fps)}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
    )

    cv2.imshow("Blurred Faces", input_array)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop = True
          

    
    
# Create face detector with LIVE_STREAM mode and callback
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    
    result_callback=print_and_blur
)

detector = FaceDetector.create_from_options(options)


# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")


while True:
    if stop:
        break
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Pass the image and timestamp to the detector
    detector.detect_async(mp_image, timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    # cv2.imshow("Face Blur", frame)
    # if cv2.waitKey(1) == 27:  # ESC to quit
    #     break

cap.release()
cv2.destroyAllWindows()
