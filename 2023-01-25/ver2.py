import cv2
import time
import os
import face_recognition

def compare_to_existing_faces(face_encoding):
    known_face_encodings = []
    known_face_names = []
    # Load existing faces and their names
    for name in os.listdir("faces"):
        if name.endswith(".jpg"):
            image = face_recognition.load_image_file("faces/" + name)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name.split(".")[0])

    # Compare the detected face to the existing faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = None
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    return name

# Open the camera
camera = cv2.VideoCapture(0)

# Set the frame width and height
camera.set(3, 640)
camera.set(4, 480)

# Wait for the camera to warm up
time.sleep(2)

# Load the haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start the camera
while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If a face is detected
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the face encoding
        face_encoding = face_recognition.face_encodings(gray[y:y+h, x:x+w])[0]

        # Compare the detected face to existing faces in the directory
        match = compare_to_existing_faces(face_encoding)

        # If there is no match
        if match is None:
            #Take the picture name from user
            picture_name = input("Enter picture name:")
            cv2.imwrite("faces/"+picture_name +".jpg", frame)
            print("New face added!")
            break
        else:
            print("Welcome back, " + match)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
