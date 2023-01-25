import cv2
import face_recognition
import os

def train_model():
    known_face_encodings = []
    known_face_names = []
    # Loop through the directory of known faces
    for dirpath, dirnames, filenames in os.walk("faces"):
        for filename in filenames:
            if filename.endswith(".jpg"):
                # Load the image
                image = face_recognition.load_image_file(os.path.join(dirpath, filename))
                # Get the face encoding
                face_encoding = face_recognition.face_encodings(image)[0]
                # Add the face encoding and name to the lists
                known_face_encodings.append(face_encoding)
                known_face_names.append(dirpath.split("/")[-1])
    return known_face_encodings, known_face_names

# Train the model with the known faces
known_face_encodings, known_face_names = train_model()

# Open the webcam
camera = cv2.VideoCapture(0)

# Set the frame width and height
camera.set(3, 640)
camera.set(4, 480)

while True:
    # Get a frame from the webcam
    ret, frame = camera.read()

    # Convert the frame to RGB
    rgb_frame = frame[:, :, ::-1]

    # Get the face encodings of the faces in the frame
    face_encodings = face_recognition.face_encodings(rgb_frame)

    # Loop through the face encodings
    for face_encoding in face_encodings:
        # Compare the face encoding to the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        match_percentage = 0.0

        # If there is a match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            match_percentage = face_recognition.face_distance([known_face_encodings[first_match_index]], face_encoding)
            match_percentage = round(1-match_percentage[0],2)

        # Draw a rectangle around the face
        (top, right, bottom, left) = face_recognition.face_locations(rgb_frame)[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name + " " + str(match_percentage*100) + "%", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
camera.release()
cv2.destroyAllWindows()
