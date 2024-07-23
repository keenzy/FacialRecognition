import cv2
import streamlit as st
from PIL import ImageColor

st.title("Face Detection using Viola-Jones Algorithm")
st.write("Instructions")
st.markdown("1)Pick the color off rectangle")
st.markdown("2)Press the button below to start detecting faces from your webcam")
st.markdown("3)Press q to quiet the webcam")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
color = st.color_picker("Pick A Color")

rectcolor = ImageColor.getcolor(color, "RGB")

scf=st.slider("Ajustez la valeur de scaleFactor?", 1.1,1.3,2.0)
mng=st.slider("Ajustez la valeur de minNeghbors?", 1,3,5)

def detect_faces():
    # Initialize the webcam
   
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scf, minNeighbors=mng)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectcolor, 2)
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        #function to save the images.
        filename = "output_image.jpg"
        cv2.imwrite(filename, frame)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()



# Add a button to start detecting faces
if st.button("Detect Faces"):
# Call the detect_faces function
            detect_faces()

