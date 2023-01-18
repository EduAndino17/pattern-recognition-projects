import cv2
import pickle
import os
import time

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Try to load the existing faces and names from a file
try:
    with open("existing_faces.pkl", "rb") as file:
        existing_faces, existing_names = pickle.load(file)
except:
    existing_faces = []
    existing_names = {}

while True:
    # Get a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop the face from the frame
        face = gray[y:y+h, x:x+w]

        # Initialize the flag for match found to False
        match_found = False
        name = None

        # Set the timer to 10 seconds
        start_time = time.time()
        
        while (time.time() - start_time) < 10:
            # Compare the face to the existing faces
            for i, existing_face in enumerate(existing_faces):
                result = cv2.matchTemplate(existing_face, face, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                # If the maximum value is above a threshold, consider it a match
                threshold = 0.8
                if max_val > threshold:
                    match_found = True
                    name = list(existing_names.keys())[list(existing_names.values()).index(i)]
                    break
            if match_found:
                break
        # If no match is found, ask for a name and save the face
        if not match_found:
            name = input("Please enter a name for the new face: ")
            existing_faces.append(face)
            existing_names[name] = len(existing_faces) - 1
            cv2.imwrite(f"{name}.jpg", face)
            print("New face added!")
        else:
            print(f"Match found! Name: {name}")
        
    # Show the webcam frame
    cv2.imshow('Webcam', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Save the existing faces and names to a file


# Save the existing faces and names to a file
with open("existing_faces.pkl", "wb") as file:
    pickle.dump((existing_faces, existing_names), file)

# Close all windows
cv2.destroyAllWindows()
