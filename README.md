# Facial-Recognition

> **Early project.** One of the first things I built, kept public as a record of where I started. Current work is on my [profile](https://github.com/samarthpaliwal).

<!-- TODO: add a screenshot as docs/screenshot.png and uncomment.
![Screenshot](docs/screenshot.png)
-->

A single-script webcam demo that detects faces and checks whether each one matches a known reference photo. It combines two approaches in one loop: an OpenCV Haar cascade for fast face detection, and the `face_recognition` library (dlib-based) to compute face encodings and compare them to the encoding of `samarth.jpg`. Matching faces are labeled with a greeting, everyone else is labeled "Unknown".

## Requirements

- Python 3.8+
- A webcam
- Python packages: `opencv-python`, `face_recognition` (which pulls in `dlib`; on most systems this needs CMake and a C++ compiler installed first, and the install can take several minutes)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install cmake
pip install opencv-python face_recognition
```

To recognize someone other than the author, replace `Facial Dection app/samarth.jpg` with a clear frontal photo of that person and change the two hard-coded references in `FaceDection.py` (`"samarth.jpg"` and the name `"Samarth"`).

## Usage

The script loads files by relative path, so run it from inside its folder:

```bash
cd "Facial Dection app"
python FaceDection.py
```

A window titled "Face Detection & Recognition" opens showing the live camera feed. Faces found by the recognizer get a green rectangle and the text `Hello, Samarth!` or `Hello, Unknown!`; faces found by the Haar cascade get a thin blue rectangle. Press `q` to quit.

If the reference photo contains no detectable face the script prints `No face found in samarth.jpg` and exits; if the camera cannot be opened it prints `ERROR: Cannot open camera`.

## How it works

1. Loads `haarcascade_frontalface_default.xml` into a `cv2.CascadeClassifier`.
2. Loads `samarth.jpg` and computes its 128-dimensional face encoding with `face_recognition.face_encodings`.
3. For every camera frame: converts to grayscale and runs `detectMultiScale` (scale factor 1.1, 5 neighbors) for the Haar boxes; separately downscales the frame to 25% and runs `face_recognition.face_locations` and `face_encodings` on it for speed.
4. Each live encoding is compared to the known encoding with `compare_faces` (default tolerance 0.6). Box coordinates are multiplied by 4 to map back to full resolution before drawing.

## Project layout

- `Facial Dection app/FaceDection.py`: the whole program (detection, recognition, drawing loop).
- `Facial Dection app/haarcascade_frontalface_default.xml`: OpenCV's pretrained frontal-face Haar cascade.
- `Facial Dection app/samarth.jpg`: reference photo used as the known identity.
