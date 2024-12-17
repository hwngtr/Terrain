import streamlit as st
import torch
import timm
from torchvision import transforms as T
from PIL import Image

def load_model(model_path):
    # Create model with correct architecture
    model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=6)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # Debug model loading
    print(f"Model keys: {state_dict.keys()}")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def predict(image, model):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.softmax(output, dim=1)
        prob, idx = torch.max(pred, 1)
        return pred.numpy(), prob.item(), idx.item()

# App title
st.title("Terrain Classification")

# Load model
model = load_model(r"C:\Users\speec\Downloads\terrain_best_model.pth")

# Display classes
classes = ['Desert', 'Forest', 'Grassland', 'Mountain', 'Ocean', 'Snow']

# Add file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    pred, confidence, class_idx = predict(image, model)
    
    st.write("Predictions:")
    sorted_preds = sorted(enumerate(pred[0]), key=lambda x: x[1], reverse=True)
    
    # Show only top predictions with meaningful confidence
    for idx, prob in sorted_preds:
        if prob > 0.2:  # Increased threshold to 20%
            st.write(f"{classes[idx]}: {prob*100:.2f}%")
    
    st.write(f"\nMost Likely: {classes[class_idx]} ({confidence*100:.2f}%)")