import cv2
import os

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

# Create a function to detect and recognize faces
def detect_and_recognize_faces():
    # Start the webcam
    cap = cv2.VideoCapture(0)

    # Create a dictionary to store the names and images of known faces
    known_faces = {}
    for file in os.listdir("./"):
        if file.endswith(".jpg"):
            name = file.split(".")[0]
            image = cv2.imread("./" + file)
            known_faces[name] = image

    # Create a LBPH face recognizer
    recognizer = cv2.LBPHFaceRecognizer_create()
    recognizer.train(list(known_faces.values()), list(known_faces.keys()))

    while True:
        # Read the webcam frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Recognize the face
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # Check if the confidence is less than 100 (a good match)
            if confidence < 100:
                name = known_faces[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                name = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            # Display the name and confidence
            cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        # Show the webcam frame
        cv2.imshow("Webcam", frame)

        # Exit the webcam if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()

# Run the function
detect_and_recognize_faces()
