import cv2
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

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the image
image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    face_encoding = face_recognition.face_encodings(gray[y:y+h, x:x+w])[0]
    # Compare the detected face to existing faces in the directory
    match = compare_to_existing_faces(face_encoding)
    if match is None:
        #Take the picture name from user
        picture_name = input("Enter picture name:")
        cv2.imwrite(picture_name +".jpg", image)
        print("New face added!")
        break
    else:
        print("Welcome back, " + match)
        break
    cv2.imwrite("detected_face.jpg", image)

# Show the original image
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
