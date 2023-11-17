import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from Cnn_Model import SimpleCNN  
import os

# Define paths
test_image_path = "dataset\\test\\test_image.jpg"
model_weights_path = "results\\models\\model_2023-08-16_12-45-42.pth"

# Data transformations (should match training preprocessing)
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the model
num_classes = 7
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load(model_weights_path))
model.eval()

# Load and preprocess the single test image
test_image = Image.open(test_image_path).convert("RGB")
test_image_transformed = data_transform(test_image)
test_image_tensor = test_image_transformed.unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    outputs = model(test_image_tensor)
    _, predicted = outputs.max(1)

# Get class label
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
predicted_label = emotion_labels[predicted.item()]

# Display the image and predicted label
plt.imshow(test_image)
plt.title(f"Predicted Emotion: {predicted_label}")
plt.axis('off')
plt.show()