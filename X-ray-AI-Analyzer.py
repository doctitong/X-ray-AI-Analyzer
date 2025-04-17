import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Load model and processor once
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("abhishek/Chest-Xray-ResNet50")
    model = AutoModelForImageClassification.from_pretrained("abhishek/Chest-Xray-ResNet50")
    model.eval()
    return processor, model

# Preprocess and predict
def analyze_xray(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits)[0]

    # Label mapping
    labels = model.config.id2label
    results = [(labels[i], float(probs[i])) for i in range(len(probs))]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]  # Top 5 conditions

# Streamlit interface
st.title("🩻 XrayGPT - Chest X-ray Assistant")

uploaded_file = st.file_uploader("Upload Chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    processor, model = load_model()
    results = analyze_xray(image, processor, model)

    st.markdown("### 🧠 Top 5 Predicted Findings:")
    for label, prob in results:
        st.write(f"**{label}** — Confidence: {prob:.2%}")
