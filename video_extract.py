from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
from deepface import DeepFace
import torch

# Load pre-trained DETR model for object detection
def load_detr_model():
    model_name = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    return processor, model

# Load an image from URL
def load_image_url(image_url):
    return Image.open(requests.get(image_url, stream=True).raw)

# Load an image from file
def load_image_file(image_path):
    return Image.open(image_path).convert("RGB")

# Display the image before processing
def show_image(image):
    image.show()

# Detect person(s) in the image and return the bounding box for the first detected person
def detect_person_bboxes(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Post-process outputs and get bounding boxes
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Extract bounding box for the person class (COCO label 1 is person)
    person_bboxes = [result["boxes"] for result in results if result["labels"] == 1]
    return person_bboxes

# Crop the image based on the bounding box of the detected person
def crop_image(image, bbox):
    left, top, right, bottom = bbox
    return image.crop((left, top, right, bottom))

# Estimate age from the cropped image using DeepFace
def estimate_age(cropped_image):
    result = DeepFace.analyze(cropped_image, actions=['age'])
    return result[0]['age']

# Main function to process the image and evaluate age
def evaluate_age(image_url):
    # Load models and image
    processor, model = load_detr_model()
    image = load_image_file(image_url)

    # Display the image before processing
    show_image(image)

    # Detect person in the image
    person_bboxes = detect_person_bboxes(image, processor, model)

    if person_bboxes:
        # Crop the image to the bounding box of the first detected person
        cropped_image = crop_image(image, person_bboxes[0])

        # Estimate the age of the person from the cropped image
        age = estimate_age(cropped_image)
        return age
    else:
        return "No person detected"

# Example usage
image_url = "image_url_here"  # Replace with your image URL
age = evaluate_age(image_url)
print(f"Estimated Age: {age}")
