import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from transformers import BlipProcessor, BlipForConditionalGeneration
from deepface import DeepFace
from PIL import Image

# Create a directory to store detected faces
output_dir = "detected_faces"
os.makedirs(output_dir, exist_ok=True)

# Load Models
print("Loading models...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
detector = MTCNN()

def extract_frames(video_path, interval=5):
    """
    Extracts frames from the video at the specified interval.
    """
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (frame_rate * interval) == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def detect_faces(frames):
    """
    Detects faces in each frame and saves cropped faces as images.
    """
    all_faces = []
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
            face_filename = f"{output_dir}/face_{i}_{face_id}.jpg"
            cv2.imwrite(face_filename, cropped_face)
            face_id += 1  # Increment face ID for uniqueness

        all_faces.append(faces_in_frame)

    print(f"Saved {face_id} detected faces in '{output_dir}' directory.")
    return all_faces

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
    all_faces = detect_faces(frames)

    print("Processing faces...")
    descriptions = []

    for faces_in_frame in all_faces:
        if faces_in_frame:
            print(f"Displaying {len(faces_in_frame)} detected faces in a frame...")
            display_faces(faces_in_frame)  # Show faces before processing

            for face in faces_in_frame:
                if method == "blip":
                    descriptions.append(describe_face_blip(face))
                elif method == "deepface":
                    descriptions.append(analyze_face_deepface(face))

    return descriptions

if __name__ == "__main__":
    video_path = "input_video.mp4"  # Change this to your video file
    method = "blip"  # Choose "blip" for natural language or "deepface" for structured output
    descriptions = process_video(video_path, interval=5, method=method)

    print("\nGenerated Descriptions:")
    for desc in descriptions:
        print(desc)
