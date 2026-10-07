import os
import platform
import threading
import urllib.request
import cv2
import mediapipe as mp
import pyvirtualcam
import tkinter as tk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#import necessary components

MODEL_PATH = 'blaze_face_full_range.tflite'
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
    )
#Global Constants
#-------------------------------------------------------------
CAM_W = 1920
CAM_H = 1080
CAM_FPS = 30

DEADZONE = 20 #Pixels center of face must move before cropped center moves to follow
#-------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    print("Downloading Face Detection model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")
#Check if model is in system. If not, download it.

detector = None
#Create detector and set it to none while model loads

def _load_detector(): #Loads detector model with settings configured
    global detector
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH)
    options=vision.FaceDetectorOptions(base_options=base_options)
    detector=vision.FaceDetector.create_from_options(options)

#run "_load_detector" on separate thread so other components can be initialized at the sametime
detector_thread=threading.Thread(target=_load_detector)
detector_thread.start()

def open_v_cam(width, height, fps):
    try: #check if there is a virtual cam to hook into
        cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR,)
        return cam, f"Virtual Camera Active: {cam.device}"
    except Exception as e: #if no v-camera exists, run in preview mode only
        return None, f"No Virtual Cam available ({e}). Running in preview-only mode."

root=tk.Tk() #setup user interface with the following configuration
root.title("Camera Config")
root.geometry("500x300")
root.attributes("-topmost", True) #UI window will always be on top

running = True #boolean to track state

def on_close(): #function to assist in closing the UI
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

preview_check = tk.Checkbutton(root, text="Show Preview Window", variable=prev_var)
preview_check.pack(pady=(15, 5))

status_label = tk.Label(root, text="Starting...", fg="gray")
status_label.pack(pady=5)


if platform.system() == "Windows": #Check what OS is currently in use and use proper video api settings
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
elif platform.system() == "Darwin":
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W) #Set video capture parameters
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

ret, t_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read from webcam")
h, w, _ = t_frame.shape #Make sure camera is readable before continuing

curr_x = None #initiate positional variables that will track center of face
curr_y = None
curr_crop_w = None #initiate positional variables that will crop video to a certain distance from the center
curr_crop_h = None

prev_open = True

vcam, status_txt = open_v_cam(w, h, CAM_FPS) #setup the virtual camera (if it exists)
prev_only = vcam is None
status_label.config(text=status_txt, fg="gray" if prev_only else "green") #change UI label color if vcam exists or not

if prev_only:
    prev_var.set(True)
    preview_check.config(state="disabled") #automatically enable preview window if no vcam exists

detector_thread.join() #Wait for detector to finish starting and join it back witht the main process

try:
    while running:
        try:
            root.update() #Process pending UI inputs
        except tk.TclError: 
            break

        PADDING = padding_var.get() #Zoom multiplier from slider (crop = face-group size * PADDING)
        SMOOTHING_FACTOR = smoothing_var.get() #Fraction of remaining distance to move

        ret, frame=cap.read() # read next frame
        if not ret: #make sure next frame can be processed
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Change color change order for media pipe (Blue Green Red to Red Green Blue)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) #wrap default image array in media pipe image
        res = detector.detect(mp_image) #Run detection on frame


        if res.detections: #build box that encloses detected faces
            min_x = min(d.bounding_box.origin_x for d in res.detections) #leftmost edge detected
            min_y = min(d.bounding_box.origin_y for d in res.detections) #topmost edge detected
            max_x = max(d.bounding_box.origin_x + d.bounding_box.width for d in res.detections) #rightmost edge detected
            max_y = max(d.bounding_box.origin_y + d.bounding_box.height for d in res.detections) #bottommost edge detected

            #center point of face box
            center_x = int((min_x+max_x)/2)
            center_y = int((min_y+max_y)/2)
            #dimmensions of face box
            group_w =  max_x - min_x
            group_h = max_y - min_y

            if curr_x is None or curr_y is None: #first detection immediately crop to face center
                curr_x = center_x
                curr_y = center_y

            #update current coordinates if 
            if abs(center_x - curr_x)> DEADZONE: #move crop center only when face has moved further than deadzone
                curr_x += (center_x-curr_x)*SMOOTHING_FACTOR 
            if abs(center_y-curr_y)>DEADZONE:
                curr_y += (center_y-curr_y)*SMOOTHING_FACTOR

            # Move the crop center toward the face center, but only once the face has
            # drifted farther than DEADZONE pixels on that axis. The move is a fraction
            # (SMOOTHING_FACTOR) of the remaining distance, so it eases in.

            #target crop width needed based off face width
            t_crop_w_x = group_w*PADDING
            #target crop width needed based off face height converted with aspect ratio
            t_crop_w_y = (group_h*PADDING)*(w/h)

            t_crop_w = int(max(t_crop_w_x, t_crop_w_y)) #use dimmension that needs larger crop
            t_crop_w = max(100, min(t_crop_w, w)) #crop between 100px and full frame width
            a_ratio = h/w #height/width ratio so crop doesnt distort

            t_crop_h = int(t_crop_w*a_ratio) #crop height matching aspect ratio

            if curr_crop_w is None: #first detection start at the target size
                curr_crop_w = t_crop_w
                curr_crop_h = t_crop_h

            #ease crop towards the target (same as position smoothing)
            curr_crop_w += (t_crop_w - curr_crop_w) * SMOOTHING_FACTOR
            curr_crop_h += (t_crop_h - curr_crop_h) * SMOOTHING_FACTOR

            #clamp topleft corner of crop so it stays in the fram
            x1 = max(0, min(int(curr_x-curr_crop_w/2),w-int(curr_crop_w)))
            y1 = max(0, min(int(curr_y-curr_crop_h/2),h-int(curr_crop_h)))

            #bottom right corner of crop
            x2 = x1 + int(curr_crop_w)
            y2 = y1 + int(curr_crop_h)

            #cut out crop region
            cropped_frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(cropped_frame, (w, h)) #scale it back to full size

        if vcam is not None: #push frame to vcam if it exists
            vcam.send(frame)
            vcam.sleep_until_next_frame() #wait so output matches fps

        #----prev window handling----
        if prev_var.get(): #checkbox is on
            if not prev_open: #if window was closed recreate it
                cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
                prev_open = True

            display_copy = cv2.resize(frame, (960, 540)) #downscale for preview
            cv2.imshow("Preview", display_copy)
            cv2.waitKey(1) #give openCV time to draw the window

            try:
                if cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE)<1: #if user closed preview with x, uncheck preview box
                    prev_var.set(False)
                    prev_open = False
            except cv2.error: #ignore if window doesnt exist
                pass

        else: #checkbox is off
            if prev_open: #close if still open
                try:
                    cv2.destroyWindow("Preview")
                except cv2.error:
                    pass
                prev_open = False
finally: #always release resources even if there is an error
    if vcam is not None:
        vcam.close()
    cap.release()
    cv2.destroyAllWindows()
    try:
        root.destroy()
    except tk.TclError: #if window was already destroyed
        pass


