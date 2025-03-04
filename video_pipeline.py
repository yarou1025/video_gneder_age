import cv2
import torch
import numpy as np
from mtcnn import MTCNN
from transformers import BlipProcessor, BlipForConditionalGeneration
from deepface import DeepFace
from PIL import Image

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
    Detects faces in each frame and returns cropped face images.
    """
    cropped_faces = []
    for frame in frames:
        results = detector.detect_faces(frame)
        for res in results:
            x, y, width, height = res['box']
            x, y = max(0, x), max(0, y)
            cropped_face = frame[y:y+height, x:x+width]
            cropped_faces.append(cropped_face)
    return cropped_faces

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
    Full pipeline to extract frames, detect faces, and analyze age & gender.
    """
    print("Extracting frames...")
    frames = extract_frames(video_path, interval)

    print("Detecting faces...")
    faces = detect_faces(frames)

    print("Processing faces...")
    descriptions = []
    for face in faces:
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
