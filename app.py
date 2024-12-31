import streamlit as st
import torch
import timm
from torchvision import transforms as T
from PIL import Image
import numpy as np
from datetime import datetime
import pickle
import os

def load_model(model_path):
    try:
        # Create RexNet model with 4 classes to match checkpoint
        model = timm.create_model(
            'rexnet_100',
            pretrained=False,
            num_classes=4,  # Match checkpoint classes
            width_mult=1.5
        )
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Initialize feedback data at start
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = {
        'images': [],
        'labels': [],
        'timestamps': []
    }

# Load existing feedback if available
feedback_file = 'feedback_data.pkl'
if os.path.exists(feedback_file):
    with open(feedback_file, 'rb') as f:
        st.session_state.feedback_data = pickle.load(f)

def predict(image, model):
    # Match training transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert image to RGB
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.softmax(output, dim=1)
        prob, idx = torch.max(pred, 1)
        return pred.numpy(), prob.item(), idx.item()
    
# App title

st.title("Terrain Classification")
model = load_model("Model.pth")
classes = ['Desert', 'Forest', 'Mountain', 'Plains']

# File upload and prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Make prediction
    predictions, prob, class_idx = predict(image, model)
    
    # Show results
    st.write("## Prediction Results")
    
    # Display all class probabilities
    for idx, (label, probability) in enumerate(zip(classes, predictions[0])):
        prob_pct = probability * 100
        st.write(f"{label}: {prob_pct:.2f}%")
    
    # Highlight top prediction
    st.write("## Top Prediction")
    st.write(f"**{classes[class_idx]}** with {prob*100:.2f}% confidence")
    
  
