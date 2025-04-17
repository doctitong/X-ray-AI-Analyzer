import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

st.title("Chest X-ray Diagnosis AI")

# Load model
@st.cache_resource
def load_model():
    model_id = "Iaroslav/chexpert-xray-classification"
    model = AutoModelForImageClassification.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    return model, processor

model, processor = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload a chest X-ray", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess and predict
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().numpy()

    # Display results
    labels = model.config.id2label
    st.subheader("AI Findings:")
    for idx, (label, prob) in enumerate(zip(labels.values(), probs)):
        st.write(f"{label}: **{prob:.2f}**")


 
