This python script creates a dynamic virtual webcam that automatically tracks, zooms, and pans the video feed so that the user(s) face is always in the direct center.
It outputs directly to OBS virtual camera which allows it to be used with Discord, Zoom, or any other video capture software.

Features:
This script uses media pipe's facial detection model to identify faces and dynamically draws a tracking box around them. A dedicated config panel allows users to adjust the zoom padding (how close the edges of the video are to the edges of the face) and tracking smoothness on the fly. It also includes a toggleable preview window that renders at a lower resolution so users can see the video being sent to their virtual camera.

Prerequisites: OBS must be installed for this program to work properly. First open OBS, then start and stop the virtual camera before closing it. After this OBS does not need to be running for the script to function.

Running the Program: If you downloaded the standalone .exe, double click it to launch the camera. The first launch will take longer so it can download the detection model from media pipe. If you downloaded the source code, install the dependencies listed in requirements.txt then launch the script in your python environment. Once the program is running, go to your video capturing software of choice and set your video input device to "OBS Virtual Camera". Make sure to disable any auto framing or studio effects to ensure the program runs smoothly.
