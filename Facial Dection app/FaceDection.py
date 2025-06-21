import os
import cv2
import face_recognition

# ---------------------------
# Load Haar Cascade (manual path)
# ---------------------------
cascade_path = "haarcascade_frontalface_default.xml"  # Must be in the same folder
face_cascade = cv2.CascadeClassifier(cascade_path)

# ---------------------------
# Load Samarth's known image for recognition
# ---------------------------
known_image = face_recognition.load_image_file("samarth.jpg")
known_encodings = face_recognition.face_encodings(known_image)

if len(known_encodings) == 0:
    print("❌ No face found in samarth.jpg")
    exit()

known_encoding = known_encodings[0]

# ---------------------------
# Start webcam
# ---------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Step 1: Detect all faces using Haar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar_faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Step 2: Recognize using face_recognition (resize for speed)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces(
            [known_encoding], face_encoding)[0]
        name = "Samarth" if match else "Unknown"

        # Scale back up face location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and name label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"Hello, {name}!", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Optional: draw rectangles from Haar face detector too
    for (x, y, w, h) in haar_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Show result
    cv2.imshow("Face Detection & Recognition", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and close
cap.release()
cv2.destroyAllWindows()
