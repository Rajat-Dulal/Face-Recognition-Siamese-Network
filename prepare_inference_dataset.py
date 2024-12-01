import os
import cv2
import shutil
import argparse

def extract_faces(input_dir, output_dir, cascade_path="haarcascade_frontalface_default.xml"):
    # Load the Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    
    if not face_cascade:
        print("Error loading Haar Cascade file.")
        return
    
    # Ensure the output directory exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Walk through the input directory
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        if os.path.isdir(folder_path):
            output_folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                    # Read the image
                    img = cv2.imread(file_path)
                    if img is None:
                        continue
                    
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    for i, (x, y, w, h) in enumerate(faces):
                        # Extract the face
                        face = img[y:y+h, x:x+w]
                        
                        # Resize to 128x128
                        face_resized = cv2.resize(face, (256, 256))
                        
                        # Save the face in the new dataset directory
                        new_file_name = f"{os.path.splitext(file_name)[0]}_face{i}.jpg"
                        new_file_path = os.path.join(output_folder_path, new_file_name)
                        cv2.imwrite(new_file_path, face_resized)

    print(f"Face extraction completed. Faces saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from images and save them in a new dataset.")
    parser.add_argument("input_dir", type=str, help="Path to the input dataset directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output dataset directory.")
    parser.add_argument(
        "--cascade", type=str, default="haarcascade_frontalface_default.xml", help="Path to the Haar cascade file."
    )

    args = parser.parse_args()
    
    # Call the function with the parsed arguments
    extract_faces(args.input_dir, args.output_dir, args.cascade)


#Run this as:
#python script_name.py DatasetPhotos Face_dataset --cascade custom_cascade.xml