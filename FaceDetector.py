import cv2
import dlib

# Initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Open the video file
video_capture = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object to save processed video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.avi', fourcc, 30, (int(video_capture.get(3)), int(video_capture.get(4))))

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = detector(gray)
    print(len(faces))
    print(faces)
    # Draw bounding boxes around detected faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Write the frame with detected faces to the output video
    # output_video.write(frame)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and output objects
video_capture.release()
# output_video.release()
cv2.destroyAllWindows()
