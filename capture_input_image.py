import cv2
import os
import shutil

def capture_image_on_keypress():
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Define the folder and image name
    folder_name = "Input_images"
    image_name = "captured.jpg"

    # Check if folder exists, delete and recreate it
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

    # Open webcam (0 is default webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'c' to capture an image, 'q' to quit, or 'Esc' to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        img= frame.copy()
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Save the image when 'c' is pressed
            if len(faces) > 0:  # Ensure a face is detected
                image_path = os.path.join(folder_name, image_name)
                cv2.imwrite(image_path, img)
                print(f"Image saved at {image_path}")
                break
            else:
                print("No face detected. Try again.")

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function
capture_image_on_keypress()
