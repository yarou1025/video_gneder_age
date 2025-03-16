import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from mtcnn import MTCNN
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Create a directory to store detected faces
output_dir = "detected_faces"
os.makedirs(output_dir, exist_ok=True)

# Load Models
print("Loading models...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
detector = MTCNN()

def extract_frames(video_path, frame_interval=1, output_folder='frames'):
    """
    Extracts frames from a video and saves them as images.

    Parameters:
    - video_path: Path to the input video file.
    - output_folder: Folder where the frames will be saved.
    - frame_interval: Save every nth frame (default is 1, meaning every frame).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    frame_names = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
             break  # Stop when video ends
        if frame_count % frame_interval == 0:  # Save every nth frame
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            frame_names.append(frame)


            # Show the frame with Matplotlib
            # plt.figure(figsize=(10, 6))
            # plt.imshow( cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # plt.axis("off")  # Hide axes
            # plt.title(f"Frame {frame_filename}")
            # plt.show()

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return frame_names

def detect_faces(frames):
    """
    Detects faces in each frame and saves cropped faces as images.
    """
    all_faces = []
    all_face_names = []
    face_id = 0  # Unique identifier for each detected face

    for i, frame in enumerate(frames):
        results = detector.detect_faces(frame)
        faces_in_frame = []

        for res in results:
            x, y, width, height = res['box']
            x, y = max(0, x), max(0, y)
            cropped_face = frame[y:y+height, x:x+width]
            faces_in_frame.append(cropped_face)

            # Save the face image
            face_filename = os.path.join(output_dir, f"face_{i}_{face_id}.jpg")
            all_face_names.append(face_filename)
            cv2.imwrite(face_filename, cropped_face)
            face_id += 1  # Increment face ID for uniqueness

        all_faces.append(faces_in_frame)

    print(f"Saved {face_id} detected faces in '{output_dir}' directory.")
    return all_faces, all_face_names

def display_faces(faces):
    """
    Displays multiple detected faces from a frame.
    """
    if not faces:
        print("No faces detected.")
        return

    fig, axes = plt.subplots(1, len(faces), figsize=(5 * len(faces), 5))
    if len(faces) == 1:
        axes = [axes]

    for ax, face in zip(axes, faces):
        ax.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        ax.axis("off")

    plt.show()

def describe_face_blip(face_image):
    """
    Uses BLIP-2 to describe the age and gender of a face image.
    """
    image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    inputs = processor(image, "Describe this person's age and gender.", return_tensors="pt")
    output = model.generate(**inputs)
    description = processor.decode(output[0], skip_special_tokens=True)
    return description

def analyze_face_deepface(face_image):
    """
    Uses DeepFace to get structured age & gender predictions.
    """
    analysis = DeepFace.analyze(face_image, actions=['age', 'gender'], enforce_detection=False)
    age = analysis[0]['age']
    gender = analysis[0]['dominant_gender']
    return f"Estimated Age: {age}, Gender: {gender.capitalize()}"

def process_video(video_path, interval=5, method="blip"):
    """
    Full pipeline to extract frames, detect faces, save faces, display faces, and analyze age & gender.
    """
    print("Extracting frames...")
    frames = extract_frames(video_path, interval)

    print("Detecting and saving faces...")
    all_faces, all_face_names = detect_faces(frames)

    print("Processing faces...")
    descriptions = []

    face_id = 0

    for i, faces_in_frame in enumerate(all_faces):
        if faces_in_frame:
            # print(f"Displaying {len(faces_in_frame)} detected faces in a frame...")
            # display_faces(faces_in_frame)  # Show faces before processing

            for face in faces_in_frame:
                if method == "blip":
                    descriptions.append(describe_face_blip(face))
                elif method == "deepface":
                    descriptions.append(analyze_face_deepface(face)+f'-face_{i}_{face_id}')
                face_id += 1

    return descriptions

if __name__ == "__main__":
    video_path = "input_video_0305.mp4"  # Change this to your video file
    method = "deepface"  # Choose "blip" for natural language or "deepface" for structured output
    descriptions = process_video(video_path, interval=25, method=method)

    print("\nGenerated Descriptions:")
    for desc in descriptions:
        print(desc)
