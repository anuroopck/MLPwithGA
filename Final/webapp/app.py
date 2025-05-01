import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import pickle

# Load best weights
with open("best_weights.pkl", "rb") as f:
    best_weights = pickle.load(f)

# Constants
INPUT_SIZE = 512
HIDDEN_SIZE = 64
OUTPUT_SIZE = 9  # number of classes

# Class labels
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
               'Pasture', 'PermanentCrop', 'Residential', 'SeaLake']

# Feature extractor
resnet = resnet18(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # Remove classification layer
resnet.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

# Prediction using genetic weights
def predict_with_weights(weights, features):
    w1 = weights[:INPUT_SIZE * HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
    b1 = weights[INPUT_SIZE * HIDDEN_SIZE : INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE]
    start = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE
    w2 = weights[start : start + HIDDEN_SIZE * OUTPUT_SIZE].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    b2 = weights[start + HIDDEN_SIZE * OUTPUT_SIZE:]
    
    h = relu(features @ w1 + b1)
    out = softmax(h @ w2 + b2)
    return out

# Streamlit UI
st.title("EuroSAT Image Classifier")

uploaded_file = st.file_uploader("Upload an RGB image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 64, 64]
    with torch.no_grad():
        features = resnet(input_tensor).squeeze().numpy()  # shape: (512,)

    probs = predict_with_weights(best_weights, features)
    top3_idx = probs.argsort()[-3:][::-1]
    
    st.markdown("### Predictions:")
    for idx in top3_idx:
        st.write(f"**{class_names[idx]}**: {probs[idx]:.4f}")

