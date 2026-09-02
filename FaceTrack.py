import os
import urllib.request
import cv2
import mediapipe as mp
import pyvirtualcam
import tkinter as tk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = 'blaze_face_full_range.tflite'
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
    )

CAM_W = 1920
CAM_H = 1080
CAM_FPS = 30

DEADZONE = 20

if not os.path.exists(MODEL_PATH):
    print("Downloading Face Detection model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

base_options=python.BaseOptions(model_asset_path=MODEL_PATH)
options=vision.FaceDetectorOptions(base_options=base_options)
detector=vision.FaceDetector.create_from_options(options)

root=tk.Tk()
root.title("Camera Config")
root.geometry("500x300")
root.attributes("-topmost", True)

running = True

def on_close():
    global running
    running = False
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

padding_var=tk.DoubleVar(value=2.5)
smoothing_var=tk.DoubleVar(value=0.08)
prev_var=tk.BooleanVar(value=True)

tk.Label(root, text="Zoom Level (lower=closer)").pack(pady=(10,0))
tk.Scale(root, variable=padding_var, from_=1.2, to=6.0, resolution=0.1, orient="horizontal").pack(fill="x", padx=20)

tk.Label(root, text="Camera Smoothing").pack()
tk.Scale(root, variable=smoothing_var, from_=0.01, to=0.20, resolution=0.01, orient="horizontal").pack(fill="x", padx=20)

tk.Checkbutton(root, text="Show Preview Window", variable=prev_var).pack(pady=15)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

ret, t_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read from webcam")
h, w, _ = t_frame.shape

curr_x = None
curr_y = None
curr_crop_w = None
curr_crop_h = None
prev_open = True

with pyvirtualcam.Camera(width=w, height=h, fps=30, fmt=pyvirtualcam.PixelFormat.BGR ) as vcam:
    while running:
        try:
            root.update()
        except tk.TclError:
            break

        PADDING = padding_var.get()
        SMOOTHING_FACTOR = smoothing_var.get()

        ret, frame=cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        res = detector.detect(mp_image)

        if res.detections:
            min_x = min(d.bounding_box.origin_x for d in res.detections)
            min_y = min(d.bounding_box.origin_y for d in res.detections)
            max_x = max(d.bounding_box.origin_x + d.bounding_box.width for d in res.detections)
            max_y = max(d.bounding_box.origin_y + d.bounding_box.height for d in res.detections)

            center_x = int((min_x+max_x)/2)
            center_y = int((min_y+max_y)/2)
            group_w =  max_x - min_x
            group_h = max_y - min_y

            if curr_x is None or curr_y is None:
                curr_x = center_x
                curr_y = center_y

            if abs(center_x - curr_x)> DEADZONE:
                curr_x += (center_x-curr_x)*SMOOTHING_FACTOR
            if abs(center_y-curr_y)>DEADZONE:
                curr_y += (center_y-curr_y)*SMOOTHING_FACTOR

            t_crop_w_x = group_w*PADDING
            t_crop_w_y = (group_h*PADDING)*(w/h)

            t_crop_w = int(max(t_crop_w_x, t_crop_w_y))
            t_crop_w = max(100, min(t_crop_w, w))
            a_ratio = h/w

            t_crop_h = int(t_crop_w*a_ratio)

            if curr_crop_w is None:
                curr_crop_w = t_crop_w
                curr_crop_h = t_crop_h

            curr_crop_w += (t_crop_w - curr_crop_w) * SMOOTHING_FACTOR
            curr_crop_h += (t_crop_h - curr_crop_h) * SMOOTHING_FACTOR

            x1 = max(0, min(int(curr_x-curr_crop_w/2),w-int(curr_crop_w)))
            y1 = max(0, min(int(curr_y-curr_crop_h/2),h-int(curr_crop_h)))
            x2 = x1 + int(curr_crop_w)
            y2 = y1 + int(curr_crop_h)

            cropped_frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(cropped_frame, (w, h))

        vcam.send(frame)
        vcam.sleep_until_next_frame()

        if prev_var.get():
            if not prev_open:
                cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
                prev_open = True

            display_copy = cv2.resize(frame, (960, 540))
            cv2.imshow("Preview", display_copy)
            cv2.waitKey(1)

            try:
                if cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE)<1:
                    prev_var.set(False)
                    prev_open = False
            except cv2.error:
                pass

        else:
            if prev_open:
                try:
                    cv2.destroyWindow("Preview")
                except cv2.error:
                    pass
                prev_open = False

cap.release()
cv2.destroyAllWindows()


